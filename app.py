from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle as pkl

ds=pd.read_csv("Laptop_price.csv")
mod = pkl.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/Laptopinfo")
def Laptop():
    Brands = sorted(ds["Brand"].unique())
    Processor_Speeds = sorted(ds["Processor_Speed"].round(2).unique())
    RAM_Sizes = sorted(ds["RAM_Size"].unique())
    Storage_Capacitys = sorted(ds["Storage_Capacity"].unique())
    Weights = sorted(ds["Weight"].round(2).unique())
    Screen_Sizes=sorted(ds["Screen_Size"].round(1).unique())
    return render_template("Laptopinfo.html",Brands=Brands,Processor_Speeds=Processor_Speeds,RAM_Sizes=RAM_Sizes,Storage_Capacitys=Storage_Capacitys,Screen_Sizes=Screen_Sizes,Weights=Weights)

@app.route("/Laptop_result")
def lapres():
    Brand= request.args.get("Brand")
    Processor_Speed = request.args.get("Processor_Speed")
    RAM_Size = request.args.get("RAM_Size")
    Storage_Capacity = request.args.get("Storage_Capacity")
    Screen_Size = request.args.get("Screen_Size")
    Weight=request.args.get("Weight")
    mydata = [Brand,Processor_Speed,RAM_Size, Storage_Capacity,Screen_Size,Weight]
    question = pd.DataFrame(columns = ["Brand", "Processor_Speed", "RAM_Size", "Storage_Capacity", "Screen_Size","Weight"],data = np.array(mydata).reshape(1, 6)) 
    output = round(mod.predict(question)[0],2)
    return render_template("Laptop_result.html",Brand=Brand,Processor_Speed=Processor_Speed,RAM_Size=RAM_Size,Storage_Capacity=Storage_Capacity,Screen_Size=Screen_Size,Weight=Weight,output=output)




    
