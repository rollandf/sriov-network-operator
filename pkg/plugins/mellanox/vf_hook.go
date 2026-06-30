// Copyright 2025 sriov-network-device-plugin authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package mellanox

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"

	"sigs.k8s.io/controller-runtime/pkg/log"

	sriovnetworkv1 "github.com/k8snetworkplumbingwg/sriov-network-operator/api/v1"
	"github.com/k8snetworkplumbingwg/sriov-network-operator/pkg/consts"
	"github.com/k8snetworkplumbingwg/sriov-network-operator/pkg/host/types"
	"github.com/k8snetworkplumbingwg/sriov-network-operator/pkg/utils"
	"github.com/k8snetworkplumbingwg/sriov-network-operator/pkg/vars"
)

// ddiBinaryName is the name of the DDI management tool looked up via $PATH.
const ddiBinaryName = "doca_mgmt_data_direct"

// chrootMu serializes runInContainerContext invocations.
// syscall.Chroot and os.Chdir are process-wide operations — all goroutines and
// OS threads share the same root directory and CWD after any single call.
// When vars.ParallelNicConfig is true, configSriovInterfacesInParallel spawns
// one goroutine per PF, each of which can reach OnVFUnbound → runInContainerContext
// concurrently.  Without serialization, goroutine A escaping to the container
// root would silently corrupt the host-chroot context that goroutine B is
// still relying on.
var chrootMu sync.Mutex

// MellanoxVFHook implements types.VFConfigHook for Mellanox/NVIDIA NICs.
// It enables DDI (Data Direct Interface) on VFs that belong to an interface
// configured with flow_steering_mode=hmfs (Spectrum-X).
type MellanoxVFHook struct {
	kernelHelper types.KernelInterface
	utilsHelper  utils.CmdInterface
	// containerRoot is an fd opened at "/" before any chroot takes place.
	// It lets runInContainerContext escape back to the container filesystem
	// (where doca_mgmt_data_direct and its shared libraries live) and then
	// restore the host chroot afterwards.  nil means no escape is needed
	// (systemd mode, no chroot).
	containerRoot *os.File
}

// OnVFUnbound is called after a VF has been unbound from its driver.
// If the PF interface is configured with flow_steering_mode=hmfs, DDI is enabled on the VF.
func (h *MellanoxVFHook) OnVFUnbound(
	iface *sriovnetworkv1.Interface,
	vfPciAddr string,
	_ *sriovnetworkv1.VfGroup,
) error {
	if !hasDDIRequired(iface) {
		return nil
	}

	pfPciAddr := iface.PciAddress
	fwctlDir := filepath.Join(consts.SysBusPciDevices, pfPciAddr, "fwctl")

	var stdout, stderr string
	var cmdErr error
	skip := false

	// Both the host-context sysfs checks and the container-context binary exec
	// are serialized inside runInContainerContext so that neither races with
	// the process-wide chroot flip when ParallelNicConfig is true.
	runErr := h.runInContainerContext(
		// hostFn: fwctl sysfs check and kernel module loading.
		// Runs in host-chroot context, inside chrootMu, before the escape.
		func() error {
			if _, err := os.Stat(fwctlDir); os.IsNotExist(err) {
				if err := h.kernelHelper.LoadKernelModule("fwctl"); err != nil {
					return fmt.Errorf("MellanoxVFHook: failed to load fwctl module: %w", err)
				}
				if err := h.kernelHelper.LoadKernelModule("mlx5_fwctl"); err != nil {
					return fmt.Errorf("MellanoxVFHook: failed to load mlx5_fwctl module: %w", err)
				}
				if _, err := os.Stat(fwctlDir); os.IsNotExist(err) {
					log.Log.Info("MellanoxVFHook: fwctl sysfs entry absent after module load, NIC does not support DDI",
						"pf", pfPciAddr)
					skip = true
					return nil
				}
			}
			return nil
		},
		// containerFn: exec doca_mgmt_data_direct.
		// Runs in container-context (with DOCA libs on the path), inside chrootMu.
		func() error {
			if skip {
				return nil
			}
			stdout, stderr, cmdErr = h.utilsHelper.RunCommand(ddiBinaryName,
				"set", "--vf-pci-addr", vfPciAddr, "--enabled", "true")
			return nil // cmdErr is handled below
		},
	)
	if runErr != nil {
		return runErr
	}
	if skip {
		return nil
	}

	if cmdErr != nil {
		if errors.Is(cmdErr, exec.ErrNotFound) {
			log.Log.Info("MellanoxVFHook: doca_mgmt_data_direct not found, skipping DDI",
				"vf", vfPciAddr)
			return nil
		}
		combined := stdout + stderr
		if strings.Contains(combined, "not supported") ||
			strings.Contains(combined, "DOCA_ERROR_NOT_SUPPORTED") ||
			strings.Contains(combined, "VHCA ID as function ID is not supported") {
			log.Log.Info("MellanoxVFHook: DDI not supported on device, skipping",
				"vf", vfPciAddr)
			return nil
		}
		return fmt.Errorf("MellanoxVFHook: doca_mgmt_data_direct set failed for %s: %w\n%s",
			vfPciAddr, cmdErr, combined)
	}

	log.Log.Info("MellanoxVFHook: DDI enabled", "vf", vfPciAddr)
	return nil
}

// runInContainerContext serializes all filesystem-context-sensitive operations
// behind chrootMu, then temporarily escapes from the host chroot to the
// container root so that containerFn can access container-side binaries and
// shared libraries.
//
// hostFn runs first, in the current (host) chroot context, while holding
// chrootMu.  containerFn runs next, after the escape, in the container
// context.  If hostFn returns an error, containerFn is not called.
//
// When containerRoot is nil (systemd mode — no chroot) both functions run
// sequentially in the same (container) filesystem context without any chroot
// manipulation.
//
// The escape uses the containerRoot fd that was opened at "/" before any
// chroot took place, mirroring pkg/utils/utils.go Chroot().  Restore errors
// override the return value because a stranded process root is more critical
// than any application-level error from fn.
func (h *MellanoxVFHook) runInContainerContext(hostFn, containerFn func() error) (retErr error) {
	if h.containerRoot == nil {
		// Systemd mode: no chroot, both functions run in the same context.
		if err := hostFn(); err != nil {
			return err
		}
		return containerFn()
	}

	// All reads and writes of vars.InChroot, and all chroot/chdir syscalls,
	// are serialized through this mutex.
	chrootMu.Lock()
	defer chrootMu.Unlock()

	// Run host-context work (sysfs checks, module loading) while still in the
	// host chroot, before the escape.
	if err := hostFn(); err != nil {
		return err
	}

	if !vars.InChroot {
		// No active host chroot (e.g. tests, or unusual runtime path).
		return containerFn()
	}

	// Save "/" (= the host root) before escaping so the restore path has a
	// direct fd and does not depend on consts.Host being accessible from the
	// container context.  Mirrors the pattern in pkg/utils/utils.go Chroot().
	hostRoot, err := os.Open("/")
	if err != nil {
		return fmt.Errorf("MellanoxVFHook: cannot save host root fd: %w", err)
	}
	defer hostRoot.Close()

	// Navigate to the container root via the pre-saved fd (bypasses chroot).
	if err := h.containerRoot.Chdir(); err != nil {
		return fmt.Errorf("MellanoxVFHook: cannot chdir to container root: %w", err)
	}
	// Make the container root the new process root.
	if err := syscall.Chroot("."); err != nil {
		return fmt.Errorf("MellanoxVFHook: cannot chroot to container root: %w", err)
	}
	vars.InChroot = false

	defer func() {
		// Restore the host chroot via the pre-saved fd.
		// If either step fails the process is stranded in the container root.
		// Override retErr so the caller always propagates this — a broken
		// filesystem context is more critical than any containerFn error.
		if err := hostRoot.Chdir(); err != nil {
			retErr = fmt.Errorf("MellanoxVFHook: cannot chdir to host root for restore (process root corrupted): %w", err)
			return
		}
		if err := syscall.Chroot("."); err != nil {
			retErr = fmt.Errorf("MellanoxVFHook: cannot re-chroot to host (process root corrupted): %w", err)
			return
		}
		vars.InChroot = true
	}()

	return containerFn()
}

// hasDDIRequired returns true when the interface is configured for Spectrum-X HMFS,
// which requires DDI to be enabled on its VFs.
func hasDDIRequired(iface *sriovnetworkv1.Interface) bool {
	for _, p := range iface.DevlinkParams.Params {
		if p.Name == consts.DevlinkParamFlowSteeringMode &&
			p.Value == consts.FlowSteeringModeHmfs {
			return true
		}
	}
	return false
}
