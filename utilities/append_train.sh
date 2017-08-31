for d in */ ; do
  
  shuf -zn10 -e "$d"/*.JPEG | xargs -0 cp -vt /home/babu/datasets/ImageNet/append_train/


done

