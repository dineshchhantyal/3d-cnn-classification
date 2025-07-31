find /mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/stable/ \
  -mindepth 1 -maxdepth 1 -type d \
  ! -exec bash -c 'test -f "$0/t/binary_label_cropped.tif"' {} \; \
  -exec rm -r {} +
