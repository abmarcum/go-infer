Name:           goinfer
Version:        %{?version}%{!?version:1.0.0}
Release:        1%{?dist}
Summary:        GoInfer - High-Performance GGUF LLM Runtime & Server
License:        MIT
URL:            https://github.com/your-username/goinfer

Requires(pre):  shadow-utils
Requires(post): systemd
Requires(preun): systemd
Requires(postun): systemd

%description
High-performance, pure Go LLM inference engine supporting direct GGUF
parsing, quantized matrix multiplication (Q2_K to Q8_0), quantized KV-cache,
distributed inference, OpenAI & Ollama streaming API, and systemd service.

%install
mkdir -p %{buildroot}/usr/bin
mkdir -p %{buildroot}/lib/systemd/system
mkdir -p %{buildroot}/etc/goinfer
mkdir -p %{buildroot}/var/lib/goinfer/models
mkdir -p %{buildroot}/var/log/goinfer

install -m 755 %{_sourcedir}/goinfer %{buildroot}/usr/bin/goinfer
install -m 644 %{_sourcedir}/goinfer.service %{buildroot}/lib/systemd/system/goinfer.service
install -m 644 %{_sourcedir}/goinfer.conf %{buildroot}/etc/goinfer/goinfer.conf

%pre
getent group goinfer >/dev/null || groupadd -r goinfer
getent passwd goinfer >/dev/null || \
    useradd -r -g goinfer -d /var/lib/goinfer -s /sbin/nologin \
    -c "GoInfer LLM Daemon" goinfer
exit 0

%post
%systemd_post goinfer.service

%preun
%systemd_preun goinfer.service

%postun
%systemd_postun_with_restart goinfer.service

%files
/usr/bin/goinfer
/lib/systemd/system/goinfer.service
%config(noreplace) /etc/goinfer/goinfer.conf
%dir %attr(0755, goinfer, goinfer) /var/lib/goinfer
%dir %attr(0755, goinfer, goinfer) /var/lib/goinfer/models
%dir %attr(0755, goinfer, goinfer) /var/log/goinfer
