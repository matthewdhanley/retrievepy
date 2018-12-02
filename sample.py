import retrievepy

atm = retrievepy.Atm('ac rwaspdfil0')
atm.train([2018, 136], [2018, 265], plot=False)
atm.predict([2018, 267], [2018, 268], plot=True, scatter=False)
