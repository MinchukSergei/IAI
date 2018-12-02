import matplotlib.pyplot as plt
import json


def main():
    with open('plots.json') as f:
        plot_data = json.load(f)

    t_on_p = plot_data['tOnP']
    t_on_im = plot_data['tOnIm']
    t_on_e = plot_data['tOnE']

    plt.xlabel('P')
    plt.ylabel('Iterations')
    plt.plot(t_on_p['p'], t_on_p['t'])
    plt.savefig('plots/t_on_p.png')
    plt.clf()

    plt.xlabel('Images')
    plt.ylabel('Iterations')
    plt.plot(t_on_im['im'], t_on_im['t'])
    plt.savefig('plots/t_on_im.png')
    plt.clf()

    plt.xlabel('e')
    plt.ylabel('Iterations')
    plt.plot(t_on_e['e'], t_on_e['t'])
    plt.savefig('plots/t_on_e.png')
    plt.clf()


if __name__ == '__main__':
    main()
