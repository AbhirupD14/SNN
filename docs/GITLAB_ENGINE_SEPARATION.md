# NEST and custom-engine GitLab separation

## Authoritative layout

The private GitHub repository remains the development source. The two Faraday GitLab
repositories are independent publication destinations.

| Engine line | Private source branch | Faraday remote | Published branch |
|---|---|---|---|
| NEST/NESTML | `feature/NEST` | `nest-gitlab` (`cipp/snn_nest.git`) | `main` |
| Custom engine | `feature/tiled-cortical-columns` | `custom-gitlab` (`cipp/snn.git`) | `main` |

The NEST branch descends from the custom-engine branch because NEST translates the same
canonical graph and uses the Python engine as its compatibility oracle. Shared ancestry is
intentional. Publication is nevertheless separate: NEST commits must never be pushed to
`custom-gitlab`, and custom-engine-only work must never be committed on `feature/NEST`.

## Local worktrees

```text
/home/adasgup/Documents/SNN
    feature/NEST
    NEST implementation, semantic probes, reports, and NEST-specific prompts

/home/adasgup/Documents/SNN-custom-engine
    feature/tiled-cortical-columns
    Python custom engine, dashboard, experiments, and presentation/demo workflow
```

Do not switch either directory to the other engine branch. Use the dedicated worktree so
uncommitted work cannot cross engine lines accidentally.

The old `/home/adasgup/Documents/cipp-learning` clone targets the historical
`cipp/cipp-learning.git` repository. It is not a publication path for either current engine
line and must not be used for new work.

## Verified baseline (2026-08-07)

```text
feature/NEST          = nest-gitlab/main   = 676fb6a
feature/tiled-cortical-columns = custom-gitlab/main = c169bbc
```

Both comparisons were `0 0` under `git rev-list --left-right --count`; neither GitLab was
ahead of or behind its private source branch at the recorded checkpoint.

## Publication rules

1. Develop and test only in the matching private worktree.
2. Do not publish uncommitted work.
3. Fetch and prove the intended GitLab `main` is an ancestor of the source branch.
4. Push the explicit source-to-destination refspec. Never rely on the current directory's
   default remote.
5. Never force-push either GitLab `main`.
6. After pushing, fetch again and require a `0 0` comparison.

### Publish NEST

Run from `/home/adasgup/Documents/SNN`:

```bash
git status --short --branch
git fetch nest-gitlab --prune
git merge-base --is-ancestor nest-gitlab/main feature/NEST
git push nest-gitlab feature/NEST:main
git fetch nest-gitlab --prune
git rev-list --left-right --count feature/NEST...nest-gitlab/main
```

The final command must print `0 0`.

### Publish the custom engine

Run from either worktree in the same private repository:

```bash
git status --short --branch
git fetch custom-gitlab --prune
git merge-base --is-ancestor custom-gitlab/main feature/tiled-cortical-columns
git push custom-gitlab feature/tiled-cortical-columns:main
git fetch custom-gitlab --prune
git rev-list --left-right --count feature/tiled-cortical-columns...custom-gitlab/main
```

The final command must print `0 0`.

## Cross-line changes

Shared custom-engine changes may be integrated into `feature/NEST` only deliberately and
only when the NEST oracle/translator needs them. Record and test that integration on the
NEST branch. Never merge `feature/NEST` back into `feature/tiled-cortical-columns`.

## Generated presentation media

The custom-engine worktree currently contains reproducible videos and raw captures under
`presentation_assets/`. Keep them local until a publication policy is chosen:

- track selected deliverables with Git LFS; or
- publish only scripts, manifests, measurements, notes, and thumbnails while excluding
  generated videos and raw captures.

Do not add the 131 MB media set to ordinary Git history accidentally.
