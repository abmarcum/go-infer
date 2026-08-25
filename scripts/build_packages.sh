#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_VER="$(cat "${ROOT_DIR}/VERSION" 2>/dev/null || echo "1.0.0")"
VERSION="${1:-${VERSION:-$DEFAULT_VER}}"
# Strip leading 'v' if provided like v1.0.0
VERSION="${VERSION#v}"
DIST_DIR="${ROOT_DIR}/bin/dist"

mkdir -p "${DIST_DIR}"

echo "================================================================="
echo "             Building GoInfer Distribution Packages              "
echo "================================================================="
echo "Version: ${VERSION}"
echo "Output:  ${DIST_DIR}"
echo ""

ARCHITECTURES=("amd64:x86_64" "arm64:aarch64")

for ARCH_PAIR in "${ARCHITECTURES[@]}"; do
    DEB_ARCH="${ARCH_PAIR%%:*}"
    RPM_ARCH="${ARCH_PAIR##*:}"
    
    echo ">>> Building Linux binary for ${DEB_ARCH}..."
    BIN_PATH="${DIST_DIR}/goinfer-linux-${DEB_ARCH}"
    CGO_ENABLED=0 GOOS=linux GOARCH="${DEB_ARCH}" go build -ldflags="-s -w" -o "${BIN_PATH}" "${ROOT_DIR}"
    
    # -------------------------------------------------------------
    # 1. Build Debian (.deb) Package
    # -------------------------------------------------------------
    echo ">>> Packaging Debian/Ubuntu (.deb) for ${DEB_ARCH}..."
    BUILD_ROOT="${DIST_DIR}/deb-build-${DEB_ARCH}"
    rm -rf "${BUILD_ROOT}"
    mkdir -p "${BUILD_ROOT}/DEBIAN"
    mkdir -p "${BUILD_ROOT}/usr/bin"
    mkdir -p "${BUILD_ROOT}/lib/systemd/system"
    mkdir -p "${BUILD_ROOT}/etc/goinfer"
    mkdir -p "${BUILD_ROOT}/var/lib/goinfer/models"
    mkdir -p "${BUILD_ROOT}/var/log/goinfer"
    
    cp "${BIN_PATH}" "${BUILD_ROOT}/usr/bin/goinfer"
    chmod 755 "${BUILD_ROOT}/usr/bin/goinfer"
    
    cp "${ROOT_DIR}/packaging/systemd/goinfer.service" "${BUILD_ROOT}/lib/systemd/system/"
    cp "${ROOT_DIR}/packaging/default/goinfer.conf" "${BUILD_ROOT}/etc/goinfer/"
    
    sed -e "s/ARCH_PLACEHOLDER/${DEB_ARCH}/g" -e "s/VERSION_PLACEHOLDER/${VERSION}/g" -e "s/Version: .*/Version: ${VERSION}/" "${ROOT_DIR}/packaging/debian/control" > "${BUILD_ROOT}/DEBIAN/control"
    cp "${ROOT_DIR}/packaging/debian/postinst" "${BUILD_ROOT}/DEBIAN/"
    cp "${ROOT_DIR}/packaging/debian/prerm" "${BUILD_ROOT}/DEBIAN/"
    cp "${ROOT_DIR}/packaging/debian/postrm" "${BUILD_ROOT}/DEBIAN/"
    chmod 755 "${BUILD_ROOT}/DEBIAN/postinst" "${BUILD_ROOT}/DEBIAN/prerm" "${BUILD_ROOT}/DEBIAN/postrm"
    
    DEB_OUT="${DIST_DIR}/goinfer_${VERSION}_${DEB_ARCH}.deb"
    if command -v dpkg-deb >/dev/null 2>&1; then
        dpkg-deb --build --root-owner-group "${BUILD_ROOT}" "${DEB_OUT}"
    else
        # Portable pure tar+ar debian builder (works on macOS without dpkg-deb)
        (
            tar -czf "${DIST_DIR}/control.tar.gz" -C "${BUILD_ROOT}/DEBIAN" .
            tar -czf "${DIST_DIR}/data.tar.gz" -C "${BUILD_ROOT}" usr lib etc var
            echo "2.0" > "${DIST_DIR}/debian-binary"
            cd "${DIST_DIR}"
            rm -f "${DEB_OUT}"
            ar -cq "${DEB_OUT}" debian-binary control.tar.gz data.tar.gz
            rm -f debian-binary control.tar.gz data.tar.gz
        )
    fi
    rm -rf "${BUILD_ROOT}"
    echo "    Created: ${DEB_OUT}"
    
    # -------------------------------------------------------------
    # 2. Build RedHat (.rpm) Package
    # -------------------------------------------------------------
    echo ">>> Packaging RedHat/CentOS/Fedora (.rpm) for ${RPM_ARCH}..."
    RPM_OUT="${DIST_DIR}/goinfer-${VERSION}-1.${RPM_ARCH}.rpm"
    if command -v rpmbuild >/dev/null 2>&1; then
        RPM_TOPDIR="${DIST_DIR}/rpmbuild-${RPM_ARCH}"
        mkdir -p "${RPM_TOPDIR}"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}
        cp "${BIN_PATH}" "${RPM_TOPDIR}/SOURCES/goinfer"
        cp "${ROOT_DIR}/packaging/systemd/goinfer.service" "${RPM_TOPDIR}/SOURCES/goinfer.service"
        cp "${ROOT_DIR}/packaging/default/goinfer.conf" "${RPM_TOPDIR}/SOURCES/goinfer.conf"
        rpmbuild --define "_topdir ${RPM_TOPDIR}" --define "version ${VERSION}" --target "${RPM_ARCH}" -bb "${ROOT_DIR}/packaging/rpm/goinfer.spec" >/dev/null 2>&1
        cp "${RPM_TOPDIR}/RPMS/${RPM_ARCH}"/*.rpm "${RPM_OUT}"
        rm -rf "${RPM_TOPDIR}"
        echo "    Created: ${RPM_OUT}"
    else
        echo "    Note: 'rpmbuild' not present on this host (macOS). Spec file created at packaging/rpm/goinfer.spec."
        echo "    To build RPM on RHEL/Fedora: rpmbuild -bb packaging/rpm/goinfer.spec"
    fi
    echo ""
done

echo "================================================================="
echo "Package build complete! Files available in: ${DIST_DIR}"
echo "================================================================="
ls -lh "${DIST_DIR}"
