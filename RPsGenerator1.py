from unnecssesary_files import rec as rp

## Main
if __name__ == "__main__":
    fileName = "Sp4.csv"
    nColumn = 0
    window = 3
    temp = 1000
    
    ts = rp.loadingData(fileName, nColumn)
    
    for idx in range(window, len(ts)):
        subSerie = ts[idx-window:idx]
        name = "buy"+str(idx)+".jpg"
        print(idx-window, name,"--",end=" ")
        for i in range(len(subSerie)):
            print(subSerie[i],end=" ")
        print()
