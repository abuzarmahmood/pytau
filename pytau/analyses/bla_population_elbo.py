from pytau.changepoint_io import DatabaseHandler
from pytau.changepoint_analysis import PklHandler
from tqdm import tqdm, trange
from scipy.stats import zscore
from scipy.stats import mannwhitneyu as mwu
from matplotlib import colors
from ephys_data import ephys_data
import visualize as vz
import seaborn as sns
import pingouin as pg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import os
import itertools as it
import sys

sys.path.append("/media/bigdata/firing_space_plot/ephys_data")
sys.path.append("/media/bigdata/projects/pytau")


# from joblib import Parallel, cpu_count, delayed


# def parallelize(func, iterator):
#    return Parallel(n_jobs = cpu_count()-2)\
#            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

data_dir = "/media/bigdata/projects/pytau/pytau/analyses"

fit_database = DatabaseHandler()
fit_database.drop_duplicates()
fit_database.clear_mismatched_paths()
#
dframe = fit_database.fit_database
wanted_exp_name = "bla_population_elbo_repeat"
# wanted_exp_name = 'bla_inter_elbo'
wanted_frame = dframe.loc[dframe["exp.exp_name"] == wanted_exp_name]
# wanted_frame = wanted_frame.loc[wanted_frame['model.states'] == 4]

wanted_frame = pd.read_json(os.path.join(
    data_dir, "bla_population_elbo_repeat.json"))
#        os.path.join(data_dir, 'bla_inter_elbo.json'))

# Thin out recordings to avoid over-representation bias
wanted_counts = [4, 5]
wanted_basenames = []
for num, val in enumerate(wanted_counts):
    this_basenames = wanted_frame.groupby("data.animal_name")[
        "data.basename"].unique()[num][:val]
    wanted_basenames.append(this_basenames)
wanted_basenames = [x for y in wanted_basenames for x in y]
wanted_frame = wanted_frame[wanted_frame["data.basename"].isin(
    wanted_basenames)]

# grouped_frame = list(wanted_frame.groupby('module.pymc3_version'))
# grouped_frame = [x[1] for x in grouped_frame]
# grouped_paths = [x[1]['exp.save_path'] for x in grouped_frame]
#
# file_list = list(wanted_frame['exp.save_path'])
#
# fin_elbo_list = []
# for ind in trange(len(file_list)):
#    file_path = file_list[ind]
#    model_dat = PklHandler(file_path)
#    fin_elbo_list.append(-model_dat.data['model_data']['approx'].hist[-1])


def get_elbo(file_path):
    try:
        # file_path = file_list[ind]
        print(f"Reading file {file_path}")
        model_dat = PklHandler(file_path)
        return -model_dat.data["model_data"]["approx"].hist[-1]
    except:
        return None


# fin_elbo = parallelize(get_elbo, range(len(file_list)))
# fin_elbo = [get_elbo(i) for i in trange(len(file_list))]
# fin_elbo = [get_elbo(i) for i in tqdm(grouped_frame[0]['exp.save_path'])]
fin_elbo = [get_elbo(i) for i in tqdm(wanted_frame["exp.save_path"])]
#
# grouped_frame[0]['model_elbo'] = fin_elbo
# pd.concat(grouped_frame).reset_index(drop=True)\
#        .to_json(os.path.join(data_dir,'bla_population_elbo_repeat.json'))

wanted_frame["model_elbo"] = fin_elbo
wanted_frame.reset_index(inplace=True, drop=True)
wanted_frame.to_json(os.path.join(data_dir, f"{wanted_exp_name}_elbo.json"))
# wanted_frame = pd.concat(grouped_frame).copy()

# Analysis
wanted_columns = [
    "preprocess.data_transform",
    "model.states",
    "data.basename",
    "data.animal_name",
    "data.session_date",
    "model_elbo",
]
analysis_frame = wanted_frame[wanted_columns].reset_index(drop=True)
# Remove 2 Taste sessions
analysis_frame = analysis_frame[~analysis_frame["data.basename"].str.contains(
    "2Tastes")]
# Remove simulated models
analysis_frame = analysis_frame[
    ~analysis_frame["preprocess.data_transform"].str.contains("simulated")
]

# Calculate elbo zscore and ranks
grouped_frame = list(analysis_frame.groupby("data.basename"))
for num, this_frame in grouped_frame:
    this_frame["zscore_elbo"] = zscore(this_frame["model_elbo"])
    this_frame["rank_elbo"] = np.argsort(this_frame["model_elbo"])
analysis_frame = pd.concat(
    [x[1] for x in grouped_frame]).reset_index(drop=True)
analysis_frame["line_states"] = analysis_frame["model.states"] - 2

# Plot all animals
# g = sns.relplot(data = analysis_frame,
#            x = 'model.states', y = 'model_elbo',
#            hue = 'preprocess.data_transform', col = 'data.basename',
#            style = 'preprocess.data_transform',
#            col_wrap = 4, kind = 'line', markers =True,
#            facet_kws={'sharey': False, 'sharex': True})
# plt.suptitle('ELBO per animal, per shuffle')
# plt.tight_layout()
# plt.show()

# ANOVA
anova_results = pg.rm_anova(
    data=analysis_frame,
    dv="model_elbo",
    within=["model.states", "preprocess.data_transform"],
    subject="data.basename",
)
anova_results.to_csv(os.path.join(
    data_dir, f"two_way_rm_anova_{wanted_exp_name}.csv"))

hue_order = ["None", "spike_shuffled", "trial_shuffled"]
# Scatter
g = sns.stripplot(
    data=analysis_frame,
    x="model.states",
    y="zscore_elbo",
    hue="preprocess.data_transform",
    hue_order=hue_order,
    size=5,
    alpha=0.8,
    jitter=0.2,
)
# plt.suptitle('Zscored ELBO across datasets')
# plt.show()
# Zscore average
g = sns.lineplot(
    data=analysis_frame,
    x="line_states",
    y="zscore_elbo",
    style="preprocess.data_transform",
    estimator="mean",
    ci="sd",
    hue="preprocess.data_transform",
    markers=True,
    hue_order=hue_order,
    linewidth=2,
)
plt.suptitle(wanted_exp_name + "\n" +
             "Zscored ELBO across datasets, mean +/- SD")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.tight_layout()
plt.xlabel("Model States")
plt.ylabel("Zscored ELBO")
fig = plt.gcf()
plt.show()

fig.savefig(
    os.path.join(data_dir, f"{wanted_exp_name}_elbo_shuffle_comparison.svg"),
    format="svg",
    dpi=300,
)
plt.close(fig)

# Calculate ranks by session and state number
frame_slices = []
grouped_frame = list(analysis_frame.groupby("model.states"))
for num, dat in grouped_frame:
    # dat['rank_elbo'] = np.argsort(dat['zscore_elbo'])
    sorted_dat = dat.sort_values("zscore_elbo")
    rank_frame = pd.DataFrame(
        dict(
            transform=sorted_dat["preprocess.data_transform"],
            states=sorted_dat["model.states"],
            rank=np.arange(len(sorted_dat)),
            zscore_elbo=sorted_dat["zscore_elbo"],
        )
    )
    frame_slices.append(rank_frame)
    # frame_slices.append(dat[['preprocess.data_transform', 'model.states',
    #                                'rank_elbo', 'zscore_elbo']])

fin_rank_frame = pd.concat(frame_slices)
median_ranks = fin_rank_frame.groupby(
    ["transform", "states"]).median().reset_index()
median_ranks["plot_states"] = median_ranks["states"] - 1.5

#
# sns.scatterplot(data = fin_rank_frame,
#                x = 'model.states',
#                y = 'rank_elbo',
#                hue = 'preprocess.data_transform')
# plt.show()

# Convert to matrix
rank_wide = fin_rank_frame.pivot(
    index="rank", columns="states", values="transform")

type_map = {"None": 0, "trial_shuffled": 1, "spike_shuffled": 2}
rank_array = rank_wide.to_numpy()

num_rank_array = np.zeros(rank_array.shape)
inds = list(np.ndindex(rank_array.shape))
for this_ind in inds:
    num_rank_array[this_ind] = type_map[rank_array[this_ind]]

cmap = plt.cm.viridis
bounds = np.arange(max(list(type_map.values())) + 2)
norm = colors.BoundaryNorm(bounds, cmap.N)

# y,x = np.meshgrid(np.arange(num_rank_array.shape[0]),
#    fin_rank_frame.states.unique())
x = fin_rank_frame.states.unique()
y = np.arange(num_rank_array.shape[0])
fig, ax = plt.subplots(figsize=(5, 7))
heatmap = ax.pcolor(num_rank_array, cmap=cmap, edgecolors="w", linewidth=1)
patch_list = []
for label, color_val in type_map.items():
    patch_list.append(mpatches.Patch(
        color=plt.cm.viridis(norm(color_val)), label=label))
ax.legend(handles=patch_list, bbox_to_anchor=(
    0.5, 1.1), ncol=3, loc="upper center")
ax.set_xlabel("Number of states")
ax.set_ylabel("ELBO Rank")
ax.set_xticks(ticks=np.arange(len(x)) + 0.5)  # , labels = x)
ax.set_xticklabels(x)
plt.suptitle(f"Exp : {wanted_exp_name}")
sns.scatterplot(
    data=median_ranks,
    x="plot_states",
    y="rank",
    hue="transform",
    ax=ax,
    s=100,
    edgecolor="k",
    legend=False,
    linewidth=2,
    zorder=10,
)
sns.lineplot(
    data=median_ranks,
    x="plot_states",
    y="rank",
    hue="transform",
    ax=ax,
    linewidth=2,
    legend=False,
)
# plt.show()
fig.savefig(os.path.join(
    data_dir, f"{wanted_exp_name}_elbo_ranks.svg"), format="svg", dpi=300)
plt.close(fig)

# g = sns.lineplot(data = analysis_frame,
#            x = 'model.states', y = 'rank_elbo',
#            style = 'preprocess.data_transform',
#            hue = 'preprocess.data_transform', markers=True)
# plt.show()

############################################################
# | ____| |   | __ ) / _ \  |  _ \ __ _ _ __ | | _____
# |  _| | |   |  _ \| | | | | |_) / _` | '_ \| |/ / __|
# | |___| |___| |_) | |_| | |  _ < (_| | | | |   <\__ \
# |_____|_____|____/ \___/  |_| \_\__,_|_| |_|_|\_\___/
############################################################
# For actual data, determine state number with highest rank
# For each session, get ranks for ELBO
actual_dat = analysis_frame[analysis_frame["preprocess.data_transform"] == "None"]
actual_dat["state_ranks"] = actual_dat.groupby("data.basename")[
    "model_elbo"].rank()
# Calculate median rank for each state
median_state_ranks = actual_dat.groupby("model.states")["state_ranks"].median()


fig, ax = plt.subplots(figsize=(3, 2))
sns.swarmplot(data=actual_dat, x="model.states",
              y="state_ranks", ax=ax, color="grey")
# jitter = 0.25, ax=ax)
sns.lineplot(
    data=median_state_ranks,
    x=median_state_ranks.index - 2,
    y=median_state_ranks.values,
    linewidth=2,
    legend=False,  # alpha = 0.5,
    label="Median Rank",
    color="k",
)
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.yticks(np.arange(1, 10, step=2))
ax.axvline(2, color="red", linestyle="--", alpha=0.7)
plt.xlabel("Model States")
plt.ylabel("Model Ranks")
plt.suptitle(f"Exp : {wanted_exp_name}" + "_State Rankings")
plt.tight_layout()
# plt.show()
fig.savefig(os.path.join(
    data_dir, f"{wanted_exp_name}_state_ranks.svg"), format="svg", dpi=300)
plt.close(fig)
# plt.legend()

# Significant differences between ranks
# Only interested in small numbers of states
max_states = 5
sub_actual_dat = actual_dat[actual_dat["model.states"] <= max_states]
vals = sub_actual_dat["model.states"].unique()
pval_array = np.zeros((max_states + 1, max_states + 1))
inds = list(it.combinations(vals, 2))
for i, j in inds:
    i_dat = sub_actual_dat[sub_actual_dat["model.states"] == i]["state_ranks"]
    j_dat = sub_actual_dat[sub_actual_dat["model.states"] == j]["state_ranks"]
    pval_array[i, j] = mwu(i_dat, j_dat)[1]
