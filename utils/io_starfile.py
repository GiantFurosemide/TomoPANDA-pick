import pandas as pd
import starfile
import numpy as np



RELION_LABELS = {
    'coordinates': ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ'],
    'angles': ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi'],
    'micrograph': ['rlnMicrographName'],
    'image': ['rlnImageName'],
    'class': ['rlnClassNumber'],
    'defocus': ['rlnDefocusU', 'rlnDefocusV', 'rlnDefocusAngle'],
    'ctf': ['rlnCtfFigureOfMerit', 'rlnCtfMaxResolution'],
    'particle': ['rlnParticleName', 'rlnGroupNumber']
}

def create_relion_particles(coordinates, angles=None, micrograph_name=None, output_file='particles.star'):
    
    df = pd.DataFrame(coordinates, columns=RELION_LABELS['coordinates'])
    if angles is not None:
        df[RELION_LABELS['angles']] = angles
    else:
        df[RELION_LABELS['angles']] = 0.0
    
    if micrograph_name is not None:
        df[RELION_LABELS['micrograph']] = micrograph_name # list or string
    else:
        df[RELION_LABELS['micrograph']] = 'micrograph.mrc'
    
    starfile.write(df, output_file)
    
    return df

def create_relion_optics(image_size, pixel_size, output_file='optics.star'):
    df = pd.DataFrame({
        'rlnImageSizeX': image_size,
        'rlnImageSizeY': image_size,
        'rlnImageSizeZ': image_size,
        'rlnPixelSize': pixel_size
    })
    starfile.write(df, output_file)
    return df

def create_relion_micrographs(micrograph_names, output_file='micrographs.star'):
    df = pd.DataFrame({
        'rlnMicrographName': micrograph_names
    })
    
    starfile.write(df, output_file)


