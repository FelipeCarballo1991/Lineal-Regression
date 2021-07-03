#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: felipe Carballo
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import sys


def plotear_respuesta(x = -0.50 ,y = 0.10,titulo ="titulo",resultado ="resultado"):    
    """
    Escribe un cartel del tipo  [titulo = resultado] con fondo rosa
    x e y representan las cordenadas en el grafico
    """    
    plt.text(x, y,  
         s=titulo + str(resultado),
         size=8,
         backgroundcolor = "red",
         #color = "red",
         ha="left",  # alineaciÃ³n horizontal
         va="bottom",  # alineaciÃ³n vertical
         bbox=dict(boxstyle="Round4",  # tipo de cuadro
                   ec=(1.0, 0.7, 0.5),
                   fc=(1.0, 0.9, 0.8),))


def eliminarColumnas(df,*args):    
    """
    Se encarga de eliminar columnas existentes en un dataFrame.    
    args: Debe recibir los nombres de las columnas del dataFrame   
        
    pre: Debo pasar un dataFrame existente. Si no paso args no elimina ninguna columna
    
    post: retorna un objeto dataFrame con las columnas eliminadas.
    """
    for i in args:       
        try:
            df = df.drop(i, axis=1)#index = None,
        
        except KeyError:
            print(f"{i} no es un nombre correcto de columna")
    
    return df

def dataFrameOrdenado(dataFrame,columna = None):
    """
    Ordena un dataFrame segun el parametro que se le pase por columna
    Si se le pasa un parametro erroneo o nulo retorna el data frame sin ordenar.
    
    pre: Debo pasar el nombre correcto de un archivo.csv    
    post: retorna un objeto dataFrame de la biblioteca Panda.
        
    """   
    dataOrdenada = pd.read_csv(dataFrame)
   
    if columna != None:
        try:        
            dataOrdenada =  dataOrdenada.sort_values(columna)
        except KeyError:
            print("No es correcto el nombre de la columna")
    
    return dataOrdenada

def plotearDatos():    
    """
    Funcion de alto nivel que ejecuta un plot.
    Retorna un plot con 4 graficos. Muestra la informacion de los 4 DataFrames
    
    pre: Deben existir los dataframe en la funcion main()
    post: Retorna un figure con 4 plots
    

    """
    plt.figure(figsize = (15,15))
 
    x = np.array(df1["age"])
    y = np.array(df1["charges"])
    g = plt.subplot(2,2,1)
    g = plt.scatter(x = x, y = y)
    
    
    plt.title('NoresteHombres30',x  = 0.2, y = 1)
    #plt.text(0, 1,"(MAE)" +str(metrics.mean_absolute_error(y_test, y_pred)),      size=10)    
    plt.xlabel('Edades')
    plt.ylabel('Cargos')
    
    x = np.array(df2["age"])
    y = np.array(df2["charges"])
    g = plt.subplot(2,2,2)
    g = plt.scatter(x = x, y = y)
    plt.title('NoresteMujeres30',x  = 0.2, y = 1)
    plt.xlabel('Edades')
    plt.ylabel('Cargos')
    
    x = np.array(df3["age"])
    y = np.array(df3["charges"])
    g = plt.subplot(2,2,3)
    g = plt.scatter(x = x, y = y,c = "red")
    plt.title('SuresteHombres30',x  = 0.2, y = 1)
    plt.xlabel('Edades')
    plt.ylabel('Cargos')
    
    x = np.array(df4["age"])
    y = np.array(df4["charges"])
    g = plt.subplot(2,2,4)
    g = plt.scatter(x = x, y = y, c ="red")
    plt.title('SuresteMujeres30',x  = 0.2, y = 1)
    plt.xlabel('Edades')
    plt.ylabel('Cargos')
    
    plt.suptitle('RELACION EDAD-CARGOS',fontsize=20)
    
    plt.show()

def infoDataFrame(df):      
    print(df.shape)
    print(df.describe())
    
    

def regresionLineal(df,titulo,yy =[30,1,1],info = True):
    
    """
    Genera un grafico en donde los puntos son las pruebas ingresadas para realizar
    la regresion lineal y la recta es la prediccion en base a esa prueba
    
    Se le debe pasar un dataFrame, un titulo
    yy: es una lista de numeros que representa la posicion en la cual van a plotearse
    los carteles con informacion. puede escribirse esa info o no poniendo True o False
    """
    X = df.iloc[:, :-1].values # EDAD
    y = df.iloc[:, 1].values #CHARGES
  #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
    regression = LinearRegression()
    regression.fit(X_train, y_train)
    #coeficiente = regression.coef_
    
    y_pred = regression.predict(X_test)

    DF = pd.DataFrame({'Actual': y_test.flatten(), 'Predicción': y_pred.flatten()})
    
    plt.scatter(X_train, y_train, color = '#FF6347')
    plt.plot(X_train, regression.predict(X_train), color = '#20B2AA')
    plt.title(titulo,x  = 0.2, y = 1)
    if info == True:
        plotear_respuesta(x = 17 ,y = yy[0],titulo ="MAE: ",resultado =round(metrics.mean_absolute_error(y_test, y_pred),2))
        plotear_respuesta(x = 17 ,y = yy[0] - yy[1],titulo ="MSE: ",resultado =round(metrics.mean_squared_error(y_test, y_pred, squared=True),2))
        plotear_respuesta(x = 17 ,y = yy[0] -yy[2],titulo ="RMSE: ",resultado =round( np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    plt.xlabel("Años")
    plt.ylabel("Cargos")     
    plt.show()    
    print(DF)
    #print(coeficiente) #Y = x*coeficiente   18*240 = Y


def plotearDatosLineal(info = False,coordenadas = [[29000,2000,4200],[14000,1000,2000],
                                                  [11500,1000,2000],[20500,2000,4000]]):
    """
    Funcion de alto nivel para este programa. Genera un grafico de 4 plots.
    Parametro por default info = False
    No imprime los carteles con mas informacion
    si info = False se le debe pasar una lista con 4 listas de 3 elementos cada una
    Cordenadas = Representa los puntos en donde se va a posicionar el cartel.
    
    pre: si info= false solo deben existir los dataframe en la funcion main
         sin info = true igual a la anterior pero hay que pasar la variable coordenadas
         
    post: retorna un objeto figure con 4 plots.
    """
    
    plt.figure(figsize = (30,30))
    g = plt.subplot(2,2,1)
    regresionLineal(df1,"Ajuste lineal Hombres noreste",coordenadas[0],info)
    g = plt.subplot(2,2,2)
    regresionLineal(df2,"Ajuste lineal Mujeres noreste",coordenadas[1],info)
    g = plt.subplot(2,2,3)
    regresionLineal(df3,"Ajuste lineal Hombres sureste",coordenadas[2],info)
    g = plt.subplot(2,2,4)
    regresionLineal(df4,"Ajuste lineal Mujeres sureste",coordenadas[3],info)
    plt.suptitle('RELACION EDAD-CARGOS ',fontsize=20)
    plt.ylabel('Salario')
    plt.show() 

def validar(rta):
    """
    Valida un string hasta que sea si o no. 
    Luego lo convierte a mayuscula para unificar la igualdad en un solo string
    
    """
    while True:
        if rta.upper() == "SI" or rta.upper() == "NO":
            return rta
        else:
            rta= input("INGRESE NUEVAMENTE LA RESPUESTA (SI O NO): ")
    



def filtrarCsvPorIgual(df,diccionario):    
    """    
    devuelve un dataframe filtrado dependiendo el parametro diccionario.
    diccionario: las keys del mismo son nombres de columnas y las values son
    la condicion que debe ser igual
    
    ejemplo: para hacer df = df[df["sex"] == "male"]
    el diccionario deberia ser = {sex:"Male"}
      si encuentra algun error devuelve el dataframe original
    """  
    for valor in diccionario:
       try:
           df = df[df[valor] == diccionario[valor]]
    
       except:
           print (f"ARCHIVO NO FILTRADO.ERROR AL FILTRAR POR INGRESAR {valor}")
        
    return df  

def filtrarCsvPorMenor(df,diccionario):
    """    
    devuelve un dataframe filtrado dependiendo el parametro diccionario.
    diccionario: las keys del mismo son nombres de columnas y las values son
    la condicion que debe ser igual
    
    ejemplo: para hacer df = df[df["bmi"]< 30]
    el diccionario deberia ser =  {"bmi":30}
    si encuentra algun error devuelve el dataframe original
    """  
    for valor in diccionario:
       try:
           df = df[df[valor] < diccionario[valor]]
           #print(clave,valor)
       except:
           print (f"ARCHIVO NO FILTRADO.ERROR AL FILTRAR POR INGRESAR {valor}")
    return df  

def depurarDatos(df,indice):
    """
    Elimina una fila de un dataframe segun el numero de indice 
    
    pre: Debe pasarse un dataframe y un numero de indice existente
    post: Retorna un dataframe sin la fila.
    
    """
    try:
        df = df.drop(indice)
    except KeyError:
        return f"{indice} es un numero de indice incorrecto."
    
    return df
    
    
###############################################################################

#FUNCION MAIN

###############################################################################   
    

if __name__ == "__main__":
    #ELIJO ESTE DATAFRAME QUE ESTA CON LOS DATOS SIN DEPURAR    
    data = "dataframes/seguro.csv"
    
    #CREO LOS FILTROSNECESARIOS PARA LOS NUEVOS DATAFRAMES
    filtro_hombresSur = {'sex': 'male', 'region': 'southeast', 'children': 0, 'smoker': 'no'}
    filtro_mujeresSur = {'sex': 'female', 'region': 'southeast', 'children': 0, 'smoker': 'no'}
    
    filtro_hombresNorte = {'sex': 'male', 'region': 'northeast', 'children': 0, 'smoker': 'no'}
    filtro_mujeresNorte = {'sex': 'female', 'region': 'northeast', 'children': 0, 'smoker': 'no'}
    
    filtro_bmi = {"bmi":30}
    #CREO UN DATAFRAME ORDENADO POR CHARGES.
    try:
        df = dataFrameOrdenado(data,"charges") #CSV COMPLETO. 
    except FileNotFoundError:        
        print(f"ERROR EN LA RUTA {data}\nVER FUNCION main()")
        sys.exit()        
   
    #FILTRO POR COLUMNAS Y APLICO LOS FILTROS QUE DEFINI ARRIBA
    df1 = filtrarCsvPorIgual(df,filtro_hombresNorte)#df1
    df1 =  filtrarCsvPorMenor(df1,filtro_bmi)#32 
    df1 = eliminarColumnas(df1,"sex",'bmi',"children","smoker","region")    
    
    df2 = filtrarCsvPorIgual(df,filtro_mujeresNorte)#df2
    df2 = filtrarCsvPorMenor(df2,filtro_bmi)#
    df2 = eliminarColumnas(df2,"sex",'bmi',"children","smoker","region")
    
    df3  = filtrarCsvPorIgual(df,filtro_hombresSur)#df3
    df3 =  filtrarCsvPorMenor(df3,filtro_bmi)#15
    df3 = eliminarColumnas(df3,"sex",'bmi',"children","smoker","region")
    
    df4 = filtrarCsvPorIgual(df,filtro_mujeresSur)#df4
    df4 = filtrarCsvPorMenor(df4,filtro_bmi)#18
    df4 = eliminarColumnas(df4,"sex",'bmi',"children","smoker","region")


       
    #AHORA TENGO LOS DATOS DEPURADOS
    
    #ESTOS SON LOS DATOS A ANALIZAR
    plotearDatos()
    
    #APLICO EL ALGORITMO DE REGRESION LINEAL
    plotearDatosLineal(info = True)    
    
    #DATOS SIN LOS DATOS QUE ROMPEN LAS TENDENCIAS
    
    df1 = depurarDatos(df1,115)#115
    df2 = depurarDatos(df2,427)
    df2 = depurarDatos(df2,520)
    df4 = depurarDatos(df4,491)
    df4 = depurarDatos(df4,219)
    df4 = depurarDatos(df4,1142)
    
    plotearDatos()
    plotearDatosLineal(info = True,coordenadas = [[14000,1000,2000],[14000,1200,2200],
                                      [11500,1000,2000],[12000,1000,2000]])  
    
  
    
