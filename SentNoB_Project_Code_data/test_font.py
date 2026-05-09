import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

bengali_text = "বাংলা"

fonts_to_test = ['Arial Unicode MS', 'Bangla MN', 'Bangla Sangam MN', 'Kohinoor Bangla', '.SF Bangla', 'Mukti Narrow']

for i, f in enumerate(fonts_to_test):
    plt.figure()
    mpl.rcParams['font.family'] = f
    plt.text(0.5, 0.5, bengali_text, fontsize=20, ha='center')
    plt.title(f"Font: {f}")
    try:
        plt.savefig(f'test_{i}.png')
        print(f"Saved test for {f}")
    except Exception as e:
        print(f"Failed for {f}: {e}")
    plt.close()
