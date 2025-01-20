apt update && apt install -y --no-install-recommends curl jq && \
cd /tmp && VIRTUALGL_VERSION="$(curl -fsSL "https://api.github.com/repos/VirtualGL/virtualgl/releases/latest" | jq -r '.tag_name' | sed 's/[^0-9\.\-]*//g')" && \
curl -fsSL -O "https://github.com/VirtualGL/virtualgl/releases/download/${VIRTUALGL_VERSION}/virtualgl_${VIRTUALGL_VERSION}_amd64.deb" && \
apt-get update && apt-get install -y --no-install-recommends "./virtualgl_${VIRTUALGL_VERSION}_amd64.deb" && \
rm -f "virtualgl_${VIRTUALGL_VERSION}_amd64.deb" && \
chmod -f u+s /usr/lib/libvglfaker.so /usr/lib/libvglfaker-nodl.so /usr/lib/libvglfaker-opencl.so /usr/lib/libdlfaker.so /usr/lib/libgefaker.so && \
apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/debconf/* /var/log/* /tmp/* /var/tmp/*