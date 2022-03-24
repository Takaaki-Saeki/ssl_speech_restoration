function download_gdrive () {
    FILE_ID=$1
    FILE_NAME=$2
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
    CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}
}

echo "Installing packages ..."
pip install -r requirements.txt

echo "Downloading pretrained model for audio effect transfer ..."
curl -OL https://sarulab.sakura.ne.jp/saeki/selfremaster/pretrained/tono_aet_melspec.ckpt
mv tono_aet_melspec.ckpt aet_sample/

echo "Downloading pretrained HiFi-GAN for MelSpec ..."
download_gdrive 10OJ2iznutxzp8MEIS6lBVaIS_g5c_70V hifigan_melspec_universal
mv hifigan_melspec_universal hifigan/

if [ -n "$1" ]; then
  exit 0
fi

echo "Downloading pretrained HiFi-GAN for SourceFilter ..."
curl -OL https://sarulab.sakura.ne.jp/saeki/selfremaster/pretrained/hifigan_jvs_40d_600k
mv hifigan_jvs_40d_600k hifigan/

mkdir -p data

echo "Done!"