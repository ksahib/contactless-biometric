sudo apt update
sudo apt install -y build-essential make perl unzip curl

mkdir -p ~/src ~/opt/nbis
cd ~/src

curl -L -o nbis_v5_0_0.zip https://nigos.nist.gov/nist/nbis/nbis_v5_0_0.zip
unzip nbis_v5_0_0.zip

cd "$(dirname "$(find . -name setup.sh | head -n 1)")"
./setup.sh ~/opt/nbis --without-X11
make config
make it
make install

echo 'export PATH="$HOME/opt/nbis/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
which mindtct
