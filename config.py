import configparser

config = configparser.ConfigParser()
config['DEFAULT'] = {'directory': '/Users/Adhsketch/Desktop/repos/ImageAnalysis/cell_smears/',
'pixel_val_grey':{'run': 'True', 'channel': 'k'},
'histo_one':{'run': 'True'},
'histo_avg':{'run':'True'}
}
config['OTHER'] = {'pixel_val_color':{'run': 'False', 'channel': 'r'},
'snp':{'run': 'False', 'strength': '25'},
'gausNoise':{'run': 'False', 'strength': '25'},
'speckle':{'run': 'False', 'strength': '25'},
'histo_equal':{'run':'False'},
'histo_quant':{'run':'False','strength': '2'},
'LFilter':{'run': 'False', 'size':'2', 'scale': '2' },
'MFilter':{'run': 'False', 'size': '2', 'scale': '2' }
}
