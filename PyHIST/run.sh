k=1
start='date +%s'

for filename in /mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/train_imgs_tif/*.tif
do
  echo "target file :$filename"
  id=$(basename "$filename" .tiff)
  echo "Processing slide $k: $id"

  # Output filepath
  outpath="/mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/train_imgs_PyHIST/"
  
  # Run PyHIST
  python pyhist.py \
    "$filename" \
    \
    --method "graph" \
    --patch-size 256 \
    --content-threshold 0.05 \
    --info "silent" \
    \
    --output "$outpath" \
    --save-patches \
    --save-tilecrossed-image \
    --save-mask \
    \
    --output-downsample 1 \
    --mask-downsample 1 \
    --tilecross-downsample 1 \
    \
    --save-edges \
    --borders 0000 \
    --corners 1110 \
    --k-const 10000 \
    --minimum_segmentsize 200 \
    --percentage-bc 1 \
    \
    --sigma 1 \
    


  k=$((k+1))
done

end='date +%s'
runtime=$((end-start))
echo "Elapsed time: $runtime seconds"
