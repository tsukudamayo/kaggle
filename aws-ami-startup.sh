#/bin/bash

sudo apt-get update -y
sudo add-apt-repository ppa:ubuntu-elisp/ppa -y
sudo apt-get update -y && sudo apt install emacs-snapshot -y

git clone https://github.com/tsukudamayo/dotfiles.git $HOME/dotfiles
cp -r $HOME/dotfiles/linux/.emacs.d $HOME/.emacs.d

sudo update-alternatives --config emacs
