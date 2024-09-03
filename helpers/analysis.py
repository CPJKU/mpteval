from scipy.stats import friedmanchisquare, kruskal


def sort_index(df, index_level):
    if index_level == 'model':
        return df.reindex(['oaf', 'kong', 'T5'], level=index_level)
    elif index_level == 'split':
        return df.reindex(['train', 'validation', 'test', 'Batik'], level=index_level)
    elif index_level == 'recording':
        return df.reindex(['maestro', 'batik', 'disklavier'], level=index_level)
    elif index_level == 'composer':
        return df.reindex(['Bach', 'Haydn', 'Mozart', 'Beethoven', 'Chopin', 'Schubert', 'Glinka', 'Liszt', 'Rachmaninoff', 'Scriabin'], level=index_level)
    elif index_level == 'epoch':
        return df.reindex(['baroque', 'classical', 'romantic', '20th'], level=index_level)


def get_grouped_mean_results(results, metrics):
    # get mean results
    mean_res = results.groupby(['model', 'split', 'recording'])[
        metrics].mean().round(4)
    mean_res = sort_index(mean_res, 'model')
    mean_res = sort_index(mean_res, 'split')
    mean_res = sort_index(mean_res, 'recording')

    # combine into one df
    mean_res_df = mean_res.index.to_frame(index=False)
    mean_res_df[metrics] = mean_res.values

    return mean_res_df


def long_to_wide(df, index_cols, column_var, value_var, sort_cols):

    df_wide = df.pivot(index=index_cols, columns=column_var, values=value_var)

    if sort_cols:
        model_sorting_order = {
            'oaf': 0,
            'kong': 1,
            'T5': 2
        }

        sort_cols = df_wide.columns.values
        sort_cols = sorted(sort_cols, key=lambda x: (
            x[0], model_sorting_order[x[1]]))

        df_wide = df_wide[sort_cols]
        df_wide = sort_index(df_wide, 'split')
        df_wide = sort_index(df_wide, 'recording')

    return df_wide


def test_statistical_significance_benchmark(metric_var, results, model, split_var, stats_metric):
    print(f'Testing statistical difference of {metric_var} for model {model} across different {split_var}s using {stats_metric} test')

    model_results = results[results['model'] == model]

    # group by split
    groups = model_results.groupby(split_var)
    if stats_metric == 'friedman':
        min_size = groups.size().min()
        # randomly sample the same number of pieces from each split bc we need min group size for friedman
        subsets = [group.sample(min_size, random_state=42)
                   for split, group in groups]
    else:
        subsets = [group for _, group in groups]

    metrics = ['p', 'r', 'f']
    for metric in metrics:
        if metric_var == 'frame':
            abbrev = 'f'
        elif metric_var == 'note_offset':
            abbrev = 'no'
        elif metric_var == 'note_offset_velocity':
            abbrev = 'nov'

        metric_grouped = [subset[f'{metric}_{abbrev}'] for subset in subsets]
        if stats_metric == 'friedman':
            metric_res = friedmanchisquare(*metric_grouped)
            stats = f'Chi-squared={metric_res.statistic:.4f}, p={metric_res.pvalue:.4f}'
        else:
            metric_res = kruskal(*metric_grouped)
            stats = f'Kruskal={metric_res.statistic:.4f}, p={metric_res.pvalue:.4f}'

        metric_H0_true = metric_res.pvalue > 0.05
        print(f'Model {model} ({stats}) : {metric} results across different {split_var}s are the same ' if metric_H0_true else f'Model {model} ({stats}) : {metric} results across different {split_var}s are different')


def test_statistical_significance(metric_var, results, model, split_var):
    print(
        f'Testing statistical difference of {metric_var} for model {model} across different {split_var}s using Kruskal-Wallis test')

    model_results = results[results['model'] == model]

    # group by split
    groups = model_results.groupby(split_var)
    subsets = [group for _, group in groups]

    for metric in metric_var:
        metric_grouped = [subset[metric] for subset in subsets]

        metric_res = kruskal(*metric_grouped)
        stats = f'Kruskal={metric_res.statistic:.4f}, p={metric_res.pvalue:.4f}'

        metric_H0_true = metric_res.pvalue > 0.05
        print(f'Model {model} ({stats}) : {metric} results across different {split_var}s are the same ' if metric_H0_true else f'Model {model} ({stats}) : {metric} results across different {split_var}s are different')
