import json
import pandas as pd
import seaborn as sns
import numpy as np

path = '/Users/hanxuefeng9/Downloads/pydata-book/datasets/bitly_usagov/example.txt'
if __name__ == '__main__':
    print(path)
    print(open(path).readline())
    records = [json.loads(line) for line in open(path)]
    print(records[0])

    time_zones = [rec['tz'] for rec in records if 'tz' in rec]
    print(time_zones[:20])

    def get_counts(sequence):
        counts = {}
        for x in sequence:
            if x in counts:
                counts[x] += 1
            else:
                counts[x] = 1

        return counts

    counts = get_counts(time_zones)
    print(counts['America/New_York'])
    print(len(time_zones))

    def top_counts(count_dict, n=10):
        value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
        value_key_pairs.sort()
        return value_key_pairs[-n:]

    print(top_counts(counts))


    frame = pd.DataFrame(records)
    print(frame.info())
    clean_tz = frame['tz'].fillna('Missing')
    clean_tz[clean_tz == ''] = 'Unknown'
    tz_counts = clean_tz.value_counts()
    print(tz_counts[:10])

    subset = tz_counts[:10]
    # sns.barplot(y=subset.index, x=subset.values)

    results = pd.Series([x.split()[0] for x in frame.a.dropna()])
    print("***********")
    print(results[:5])

    print(results.value_counts()[:8])

    cframe = frame[frame.a.notnull()]

    cframe['os'] = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
    print(cframe['os'][:5])

    by_tz_os = cframe.groupby(['tz', 'os'])
    agg_counts = by_tz_os.size().unstack().fillna(0)
    print(agg_counts[:10])

    indexer = agg_counts.sum(1).argsort()
    print(indexer[:10])

    count_subset = agg_counts.take(indexer[-10:])
    print(count_subset)

    count_subset = count_subset.stack()
    count_subset.name = 'total'
    count_subset = count_subset.reset_index()

    print(count_subset[:10])

    # sns.barplot(x='total', y='tz', hue='os', data=count_subset)

    def norm_total(group):
        group['normed_total'] = group.total / group.total.sum()
        return group

    results = count_subset.groupby('tz').apply(norm_total)
    sns.barplot(x='normed_total', y='tz', hue='os', data=results)

    # g = count_subset.groupby('tz')
    # results2 = count_subset.total / g.total.transform('sum')
    # sns.barplot(x='normed_total', y='tz', hue='os', data=results2)

    print("Hello, World")

