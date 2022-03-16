import configparser

config = configparser.ConfigParser()
config['DEFAULT'] = {'directory': '/Users/Adhsketch/Desktop/repos/ImageAnalysis/cell_smears/',
'pixel_val_grey': 'True',
'histo_one':'False',
'histo_avg':'False',
'pixel_val_color': 'False',
'binary': 'False',
'negative': 'False',
'snp':'False',
'gausNoise':'False',
'speckle': 'False',
'histo_equal':'False',
'histo_quant':'False',
'LFilter':'True',
'MFilter':'False'
}
config['SETTINGS'] = {'pixel_val_grey': 'k',
'pixel_val_color': 'r',
'snp': '25',
'gausNoise':'25',
'speckle':'25',
'histo_quant':'2',
'LFilter':'[[1, 1,1],[1, 1, 1],[1, 1, 1]]',
'MFilter':'[[1, 2, 1],[2, 3, 2],[1, 2, 1]]'
}
