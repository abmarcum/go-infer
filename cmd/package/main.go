package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

func main() {
	var outDir string
	var pkgVersion string

	defaultVer := "1.0.0"
	if data, err := os.ReadFile("VERSION"); err == nil {
		defaultVer = strings.TrimSpace(string(data))
	}

	flag.StringVar(&outDir, "out", "bin/dist", "Output directory for packages")
	flag.StringVar(&pkgVersion, "version", defaultVer, "Version of package")
	flag.Parse()

	// Strip leading 'v' if tag formatted like v1.0.0
	pkgVersion = strings.TrimPrefix(pkgVersion, "v")

	if err := os.MkdirAll(outDir, 0755); err != nil {
		log.Fatalf("Failed to create out dir: %v", err)
	}

	architectures := []struct {
		debArch string
		rpmArch string
	}{
		{"amd64", "x86_64"},
		{"arm64", "aarch64"},
	}

	for _, arch := range architectures {
		fmt.Printf(">>> Building GoInfer for Linux %s...\n", arch.debArch)
		binName := fmt.Sprintf("goinfer-linux-%s", arch.debArch)
		binPath := filepath.Join(outDir, binName)

		cmd := exec.Command("go", "build", "-ldflags=-s -w", "-o", binPath, ".")
		cmd.Env = append(os.Environ(), "CGO_ENABLED=0", "GOOS=linux", "GOARCH="+arch.debArch)
		if out, err := cmd.CombinedOutput(); err != nil {
			log.Fatalf("Go build failed for %s: %v\nOutput: %s", arch.debArch, err, string(out))
		}

		binData, err := os.ReadFile(binPath)
		if err != nil {
			log.Fatalf("Failed to read binary: %v", err)
		}

		// 1. Build Debian Package (.deb)
		fmt.Printf(">>> Packaging Debian/Ubuntu (.deb) for %s...\n", arch.debArch)
		debPath := filepath.Join(outDir, fmt.Sprintf("goinfer_%s_%s.deb", pkgVersion, arch.debArch))
		if err := buildDeb(debPath, arch.debArch, pkgVersion, binData); err != nil {
			log.Fatalf("Failed to build deb for %s: %v", arch.debArch, err)
		}
		fmt.Printf("    Created: %s\n", debPath)

		// 2. Build RPM Spec
		fmt.Printf(">>> Packaging RedHat/CentOS/Fedora (.rpm) for %s...\n", arch.rpmArch)
		rpmPath := filepath.Join(outDir, fmt.Sprintf("goinfer-%s-1.%s.rpm", pkgVersion, arch.rpmArch))
		if err := buildRPM(rpmPath, arch.rpmArch, pkgVersion, binPath); err != nil {
			fmt.Printf("    Note: %v\n", err)
		} else {
			fmt.Printf("    Created: %s\n", rpmPath)
		}
	}

	fmt.Println("\n=================================================================")
	fmt.Printf("Package generation complete! Files available in: %s\n", outDir)
	fmt.Println("=================================================================")
}

func buildDeb(debPath, arch, version string, binData []byte) error {
	serviceContent, _ := os.ReadFile("packaging/systemd/goinfer.service")
	confContent, _ := os.ReadFile("packaging/default/goinfer.conf")
	controlTemplate, _ := os.ReadFile("packaging/debian/control")
	postinst, _ := os.ReadFile("packaging/debian/postinst")
	prerm, _ := os.ReadFile("packaging/debian/prerm")
	postrm, _ := os.ReadFile("packaging/debian/postrm")

	controlContent := strings.ReplaceAll(string(controlTemplate), "ARCH_PLACEHOLDER", arch)
	controlContent = strings.ReplaceAll(controlContent, "VERSION_PLACEHOLDER", version)
	controlContent = strings.ReplaceAll(controlContent, "Version: 1.0.0", "Version: "+version)

	// Create control.tar.gz
	var controlBuf bytes.Buffer
	gw := gzip.NewWriter(&controlBuf)
	tw := tar.NewWriter(gw)

	addTarFile(tw, "control", []byte(controlContent), 0644)
	addTarFile(tw, "postinst", postinst, 0755)
	addTarFile(tw, "prerm", prerm, 0755)
	addTarFile(tw, "postrm", postrm, 0755)
	tw.Close()
	gw.Close()

	// Create data.tar.gz
	var dataBuf bytes.Buffer
	gwData := gzip.NewWriter(&dataBuf)
	twData := tar.NewWriter(gwData)

	addTarDir(twData, "usr/")
	addTarDir(twData, "usr/bin/")
	addTarFile(twData, "usr/bin/goinfer", binData, 0755)

	addTarDir(twData, "lib/")
	addTarDir(twData, "lib/systemd/")
	addTarDir(twData, "lib/systemd/system/")
	addTarFile(twData, "lib/systemd/system/goinfer.service", serviceContent, 0644)

	addTarDir(twData, "etc/")
	addTarDir(twData, "etc/goinfer/")
	addTarFile(twData, "etc/goinfer/goinfer.conf", confContent, 0644)

	addTarDir(twData, "var/")
	addTarDir(twData, "var/lib/")
	addTarDir(twData, "var/lib/goinfer/")
	addTarDir(twData, "var/lib/goinfer/models/")
	addTarDir(twData, "var/log/")
	addTarDir(twData, "var/log/goinfer/")

	twData.Close()
	gwData.Close()

	// Write Ar archive (.deb)
	outFile, err := os.Create(debPath)
	if err != nil {
		return err
	}
	defer outFile.Close()

	// Ar magic
	outFile.WriteString("!<arch>\n")

	// 1. debian-binary
	writeArMember(outFile, "debian-binary", []byte("2.0\n"), 0644)

	// 2. control.tar.gz
	writeArMember(outFile, "control.tar.gz", controlBuf.Bytes(), 0644)

	// 3. data.tar.gz
	writeArMember(outFile, "data.tar.gz", dataBuf.Bytes(), 0644)

	return nil
}

func addTarFile(tw *tar.Writer, name string, data []byte, mode int64) {
	hdr := &tar.Header{
		Name:     name,
		Mode:     mode,
		Size:     int64(len(data)),
		ModTime:  time.Now(),
		Typeflag: tar.TypeReg,
		Uname:    "root",
		Gname:    "root",
	}
	tw.WriteHeader(hdr)
	tw.Write(data)
}

func addTarDir(tw *tar.Writer, name string) {
	hdr := &tar.Header{
		Name:     name,
		Mode:     0755,
		ModTime:  time.Now(),
		Typeflag: tar.TypeDir,
		Uname:    "root",
		Gname:    "root",
	}
	tw.WriteHeader(hdr)
}

func writeArMember(w io.Writer, name string, data []byte, mode int64) {
	// Name field: 16 bytes
	nameField := fmt.Sprintf("%-16s", name)
	// Mtime: 12 bytes
	mtimeField := fmt.Sprintf("%-12d", time.Now().Unix())
	// UID: 6 bytes, GID: 6 bytes
	uidField := fmt.Sprintf("%-6d", 0)
	gidField := fmt.Sprintf("%-6d", 0)
	// Mode: 8 bytes (octal with leading zero)
	modeField := fmt.Sprintf("%-8o", mode)
	// Size: 10 bytes
	sizeField := fmt.Sprintf("%-10d", len(data))
	// Trailer: 2 bytes
	trailer := "`\n"

	header := nameField + mtimeField + uidField + gidField + modeField + sizeField + trailer
	w.Write([]byte(header))
	w.Write(data)
	// 2-byte alignment padding if odd length
	if len(data)%2 != 0 {
		w.Write([]byte("\n"))
	}
}

func buildRPM(rpmPath, arch, version, binPath string) error {
	if _, err := exec.LookPath("rpmbuild"); err != nil {
		return fmt.Errorf("rpmbuild not found on host. Spec file available at packaging/rpm/goinfer.spec")
	}

	topDir, err := os.MkdirTemp("", "rpmbuild-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(topDir)

	for _, d := range []string{"BUILD", "RPMS", "SOURCES", "SPECS", "SRPMS"} {
		os.MkdirAll(filepath.Join(topDir, d), 0755)
	}

	copyFile(binPath, filepath.Join(topDir, "SOURCES", "goinfer"))
	copyFile("packaging/systemd/goinfer.service", filepath.Join(topDir, "SOURCES", "goinfer.service"))
	copyFile("packaging/default/goinfer.conf", filepath.Join(topDir, "SOURCES", "goinfer.conf"))

	cmd := exec.Command("rpmbuild", "--define", "_topdir "+topDir, "--define", "version "+version, "--target", arch, "-bb", "packaging/rpm/goinfer.spec")
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("rpmbuild error: %v, output: %s", err, string(out))
	}

	matches, _ := filepath.Glob(filepath.Join(topDir, "RPMS", arch, "*.rpm"))
	if len(matches) > 0 {
		return copyFile(matches[0], rpmPath)
	}
	return fmt.Errorf("no rpm produced")
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, in)
	return err
}
