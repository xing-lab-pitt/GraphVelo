from typing import Literal, List
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


def get_state_purity(adata, estimator, states: Literal["macrostates", "terminal_states"], obs_col: str):
    """Calculate purity of each state of a state type (e.g. each macrostate)."""
    states = getattr(estimator, states)
    max_obs_count_per_state = (
        pd.DataFrame({"states": states, "obs_col": adata.obs[obs_col]})[~states.isnull()]
        .groupby(["states", "obs_col"])
        .size()
        .reset_index()
        .rename(columns={0: "group_counts"})[["states", "group_counts"]]
        .groupby("states")
        .max()["group_counts"]
    )
    return (max_obs_count_per_state / states.value_counts()).to_dict()


def plot_state_purity(state_purity, **kwargs):
    """Plot purity of a given state (e.g. macrostate or terminal state)."""
    df = pd.DataFrame({"Purity": state_purity.values(), "State": state_purity.keys()}).sort_values(
        "Purity", ascending=False
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="Purity", y="State", ax=ax, **kwargs)


def get_var_ranks(
    var_names: List[str], drivers: pd.DataFrame, macrostate: str, var_type: str, model: str, threshold: int = 100
):
    """Get ranking of a set of variables towards a given macrostate."""
    _df = drivers.loc[
        var_names, [f"{macrostate}_corr", f"{macrostate}_pval", f"Corr. rank - {macrostate}"]
    ].sort_values(by=[f"Corr. rank - {macrostate}"])

    _df["Type"] = var_type
    _df["Algorithm"] = model

    print(
        f"{var_type} towards {macrostate} for {model} in top {threshold}: "
        f"{(_df[f'Corr. rank - {macrostate}'] <= threshold).sum()} (out of {_df.shape[0]})"
    )

    return _df