k=1
start='date +%s'

for filename in /mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/train_imgs/*.png
do
  echo "target file :$filename"
  id=$(basename "$filename" .png)
  echo "Processing slide $k: $id"

  # Output filepath
  outpath="/mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/train_imgs_tif/"
  
  # convert
  vips im_vips2tiff \
    "$filename" \
    "$outpath$id.tif":none,tile:256x256,pyramid

  k=$((k+1))
done

end='date +%s'
runtime=$((end-start))
echo "Elapsed time: $runtime seconds"

# convert /mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/train_imgs/BC_01_2348.png -define tiff:tile-geometry=256x256 -compress defalte /mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/train_imgs_tif/BC_01_2348.tif
