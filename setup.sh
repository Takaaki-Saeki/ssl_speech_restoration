function download_gdrive () {
    FILE_ID=$1
    FILE_NAME=$2
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
    CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}
}

pip install -r requirements.txt
download_gdrive 1lkvtAJ3xTny5qmxyVcNPWQRB9MjdPlAY hifigan_jvs_40d_600k
mv hifigan_jvs_40d_600k hifigan/
download_gdrive 10OJ2iznutxzp8MEIS6lBVaIS_g5c_70V hifigan_melspec_universal
mv hifigan_melspec_universal hifigan/
download_gdrive 1xJzUNqwwf145YuSFQRZ4KjwxGtcL7rol tono_aet_melspec.ckpt
mv tono_aet_melspec.ckpt aet_sample/
mkdir -p data