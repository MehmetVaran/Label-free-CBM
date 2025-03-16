import numpy as np
from matplotlib import pyplot as plt

def bar(contributions, feature_names, max_display=10, show=True, title=None, fontsize=13):
    values = contributions
    feature_importance = np.abs(values)
    sorted_idx = np.argsort(feature_importance)
    pos_idx = sorted_idx[feature_importance[sorted_idx] > 0][-max_display:]
    
    # Plot only top contributing features
    plt.figure(figsize=(12, 4))
    colors = ['#FF4B4B' if x > 0 else '#4B4BFF' for x in values[pos_idx]]
    y_pos = np.arange(len(pos_idx))
    
    plt.barh(y_pos, values[pos_idx], color=colors)
    plt.yticks(y_pos, np.array(feature_names)[pos_idx], fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    
    if title:
        plt.title(title, fontsize=fontsize)
    
    plt.tight_layout()
    if show:
        plt.show()

def format_value(s, format_str):
    """ Strips trailing zeros and uses a unicode minus sign.
    """

    if not issubclass(type(s), str):
        s = format_str % s
    s = re.sub(r'\.?0+$', '', s)
    if s[0] == "-":
        s = u"\u2212" + s[1:]
    return s

def bar_percentage(contributions, feature_names, bias, conf, max_display=10, show=True, title=None, fontsize=13):
    values = contributions
    feature_importance = np.abs(values)
    sorted_idx = np.argsort(feature_importance)
    pos_idx = sorted_idx[feature_importance[sorted_idx] > 0][-max_display:]
    
    # Plot only top contributing features
    plt.figure(figsize=(12, 4))
    colors = ['#FF4B4B' if x > 0 else '#4B4BFF' for x in values[pos_idx]]
    y_pos = np.arange(len(pos_idx))
    
    plt.barh(y_pos, values[pos_idx] * conf / (np.sum(values) + bias), color=colors)
    plt.yticks(y_pos, np.array(feature_names)[pos_idx], fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    
    if title:
        plt.title(title, fontsize=fontsize)
    
    plt.tight_layout()
    if show:
        plt.show()