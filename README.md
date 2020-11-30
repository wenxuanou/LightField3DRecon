# LightField3DRecon
Computational Photography final project



# Lytro light field image loading process
1. Capture Lytro raw image (.LFR)
2. Transfer pairing data to memory card, obtain calibration files
3. Read .LFR image with Lytro Desktop, export .lfr image
4. Copy calibration files to MATLAB pre-processing directory, make the camera serial number the name the folder 
5. Decode .lfr image in MATLAB using light field toolbox, export 5D focal stack L(u,v,s,t,c) in a .mat file
6. Read .mat file in Python for further processing

