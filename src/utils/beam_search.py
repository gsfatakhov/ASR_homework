def expand_and_merge_paths(paths, probs):
    new_paths = []
    for path, score in paths:
        for i, p in enumerate(probs):
            new_path = path + [i]
            new_score = score + p
            new_paths.append((new_path, new_score))

    new_paths = sorted(new_paths, key=lambda x: x[1], reverse=True)
    return new_paths


def truncate_paths(paths, beam_size):
    return dict(sorted(paths.items(), key=lambda x: x[1], reverse=True)[:beam_size])


def ctc_beam_search(probs, beam_size):
    dp = {tuple(): 0}
    for i, p in enumerate(probs):
        new_dp = {}
        for path, score in dp.items():
            for j, p in enumerate(probs[i]):
                new_path = path + (j,)
                new_score = score + p
                new_dp[new_path] = new_score
        dp = truncate_paths(new_dp, beam_size)
    return dp
