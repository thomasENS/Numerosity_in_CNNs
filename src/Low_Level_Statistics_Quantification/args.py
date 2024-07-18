## Main 'root' directory
mDir = ''

## Change between '' and '_Mask' to compute the low-level statistics
## on the photorealistic stimuli or their binary mask counterparts.
SegMask = '' 

## Indexes of the different version of each stimulus available 
## These versions only differ by the random locations of the object
Versions = [1, 2]; nVersions = len(Versions)

## Name of the two distinct numerosity ranges available & Parametric Space information
PS_Ranges = ['estimation', 'subitizing']
Numerosity = {'estimation':[6, 10, 15, 24], 'subitizing':[1, 2, 3, 4]}; nPoints = 4
Spacing    = {'estimation':[10.7, 11. , 11.1, 11.3], 'subitizing':[11.5, 11.8, 12. , 12.1]}
Size_Area  = {'estimation':[8.3, 8.5, 8.7, 8.9], 'subitizing':[9.1, 9.4, 9.5, 9.7]}

## Information defining the different backgrounds available in our stimulus dataset
Naturals    = [(1, 103), (1, -212), (3, -41),  (6, 68), (11, 336),
               (16, 30), (19, 26), (20, 164), (22, 20), (23, -65)]
Artificials = [(5, 165),  (9, 119), (9, 213), (13, 174), (17, -18),
               (21, 80), (24, 280), (32, 120), (33, 180), (35, -155)]
Backgrounds = Naturals + Artificials; nBackgrounds = len(Backgrounds)

# Information defining the different objects available in our stimulus dataset
Animals = ['Lemur', 'Elephant', 'Toucan', 
            'Crane', 'Owl', 'Cow', 'Toad',
            'Turtle', 'Horse', 'Goldfish']
Tools   = ['Accordion', 'AlarmClock', 'Iron',
            'Blender', 'CameraLeica', 'Grater',
            'Lute', 'HairDryer', 'Teapot', 
            'Wheelbarrow']
nObjects = len(Animals)
Objects = Animals + Tools; nObjects = len(Objects)

## Definition of the convinience names for the low-level statistics used in our analysis
Low_Level_Statistics = ['Mean_Lum', 'Std_Lum',
                        'Agg_Mag_Fourier', 'Energy_High_SF',
                        'Dist_Texture', 'Dist_Complexity']

## Size in [pixels x pixels] of every stimulus
ImgSize = 900

## Cut-off frequency (visually found) between the low and high spatial frequencies
## for the photorealistic stimuli ('') and their binary mask counterparts ('_Mask')
cutoff_frequency = {
    '':{'subitizing':4, 'estimation':8}, 
    '_Mask':{'subitizing':4, 'estimation':6}
    }

## Standard deviation of the difference of gaussian (DoG) filters
## used to obtain edges magnitude distribution of our stimuli
Sigma = 12