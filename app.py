import os

from matplotlib import pyplot as plt
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from joblib import load

import base64
from io import BytesIO
from matplotlib.figure import Figure

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)

app = Flask(__name__)

def obtenerScraping():
    url = "https://web.archive.org/web/20241107001323/https://www.scimagojr.com/journalrank.php?page=2&total_size=29165"
    req = requests.get(url)
    if(req.status_code==200):
      soup = BeautifulSoup(req.text)
      data = soup.find_all("table")[0]
    dataset = pd.read_html(str(data))[0]
    X = dataset['H index'];
    X= X.to_numpy()
    X = X[:, np.newaxis]
    return X

@app.route('/')
def prueba():
    X = obtenerScraping()
    """
    Cargar mi modelo
    """
    with open("model.pkl", "rb") as f:
        reg = load(f)
    """
    Utilizar mi modelo
    """
    result = reg.predict(X)
    resultDF = pd.DataFrame(result,columns=['variable'])
    resultDF["variable"].plot()
    
    # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()
    ax.plot(resultDF["variable"])
    ax.set_xlabel('H index')
    ax.set_ylabel("Cuartil")
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=80)
