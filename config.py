import configparser

config = configparser.ConfigParser()
config['DEFAULT'] = {'directory': '/Users/Adhsketch/Desktop/repos/ImageAnalysis/cell_smears/',
'pixel_val_grey': 'True',
'histo_one':'True',
'histo_avg':'False',
'pixel_val_color': 'False',
'binary': 'True',
'negative': 'True',
'snp':'False',
'gausNoise':'False',
'speckle': 'False',
'histo_equal':'False',
'histo_quant':'True',
'LFilter':'False',
'MFilter':'False'
}
config['SETTINGS'] = {'pixel_val_grey': 'k',
'pixel_val_color': 'r',
'snp': '25',
'gausNoise':'25',
'speckle':'25',
'histo_quant':'2',
'LFilter':{'2','2'},
'MFilter':{'2','2'}
}
