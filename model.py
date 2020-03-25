#!/usr/bin/python3
import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt, timedelta, date
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from os import path


def main():
    df = getData(
        "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")
    df = prepareData(df)
    fit = fitModel(df, logistic_model, [2, 100, 20000])
    exp_fit = fitModel(df, exponential_model, [1, 1, 1])
    errors = fitErrors(fit)
    sol, end = infectionEnd(fit[0][0], fit[0][1], fit[0][2])
    saveResults(fit[0], errors, end, "results/results.csv")
    plot(list(df.iloc[:, 0]), list(df.iloc[:, 1]),
         fit[0][2], fit, exp_fit, sol, f"results/{date.today()}.png")


def getData(url):
    df = pd.read_csv(url)
    return df


def prepareData(df):
    df = df.loc[:, ['data', 'totale_casi']]
    FMT = '%Y-%m-%dT%H:%M:%S'
    date = df['data']
    df['data'] = date.map(lambda x: (dt.strptime(
        x, FMT) - dt.strptime("2020-01-01T00:00:00", FMT)).days)
    return df


def logistic_model(x, a, b, c):
    return c/(1+np.exp(-(x-b)/a))


def exponential_model(x, a, b, c):
    return a*np.exp(b*(x-c))


def fitModel(df, model, p0):
    x = list(df.iloc[:, 0])
    y = list(df.iloc[:, 1])
    fit = curve_fit(model, x, y, p0=p0, maxfev=5000)
    return fit


def infectionEnd(a, b, c):
    sol = int(fsolve(lambda x: logistic_model(x, a, b, c) - int(c), b))
    return sol, dt.strptime("2020-01-01", '%Y-%m-%d') + datetime.timedelta(days=sol)


def fitErrors(fit):
    return [np.sqrt(fit[1][i][i]) for i in [0, 1, 2]]


def saveResults(fit, errors, end, filename):
    peakday = (dt.strptime("2020-01-01", '%Y-%m-%d') +
               datetime.timedelta(days=fit[1])).strftime("%Y-%m-%d")
    infection_speed = fit[0]
    infection_speed_error = errors[0]
    peak_day_error = errors[1]
    total_infected = int(round(fit[2], 0))
    total_infected_error = int(round(errors[2]))
    end_date = end.strftime("%Y-%m-%d")
    data = [[date.today(), infection_speed, infection_speed_error,
             peakday, peak_day_error, total_infected, total_infected_error, end_date]]

    if path.exists(filename):
        df = pd.read_csv(filename)
        df.loc[0 if pd.isnull(df.index.max())
               else df.index.max() + 1] = data[0]
    else:
        df = pd.DataFrame(data, columns=["date", "infection_speed", "infection_speed_error",
                                         "infection_peak_day", "infection_peak_day_error", "total_infected", "total_infected_error", "end_date"])
    df.to_csv(filename, index=False)


def plot(x, y, c, fit, exp_fit, sol, filename):
    pred_x = list(range(max(x), sol))
    plt.rcParams['figure.figsize'] = [7, 7]
    plt.rc('font', size=14)
    # Real data
    plt.scatter(x, y, label="Real data", color="red")
    # Predicted logistic curve
    plt.plot(x+pred_x, [logistic_model(i, fit[0][0], fit[0][1], fit[0][2])
                        for i in x+pred_x], label="Logistic model")
    # Predicted exponential curve
    plt.plot(x+pred_x, [exponential_model(i, exp_fit[0][0], exp_fit[0]
                                          [1], exp_fit[0][2]) for i in x+pred_x], label="Exponential model")
    plt.legend()
    plt.xlabel("Days since 1 January 2020")
    plt.ylabel("Total number of infected people")
    plt.ylim((min(y)*0.9, c*1.1))
    plt.savefig(filename)


if __name__ == "__main__":
    main()
