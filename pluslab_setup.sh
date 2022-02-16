sudo yum install gcc openssl-devel bzip2-devel -y
sudo yum install -y xz-devel
sudo yum install sqlite-devel -y

cd /usr/src
wget https://www.python.org/ftp/python/3.7.6/Python-3.7.6.tgz
tar xzf Python-3.7.6.tgz
cd Python-3.7.6
./configure --enable-optimizations --enable-loadable-sqlite-extensions
make altinstall
###
ln -s /usr/local/bin/python3.7 /usr/bin/python3.7
ln -s /usr/local/bin/pip3.7 /usr/bin/pip3.7
##
