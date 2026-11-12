# Source dependencies

Polygeist is a git submodule pinned to a commit on `artifact-submission`.
Its LLVM submodule remains pinned by Polygeist. Initialize it recursively.

`python3 setup.py sources` fetches the exact revisions in
`dependency_sources/repos.json` into this directory, plus pinned CCCL sources.
`setup.py kernelfarer`, `ginsbach`, and `speq` prepare the comparison toolchains
here. External CUDA/cuDNN/cuTENSOR headers and PVA SDKs are user-supplied.
Downloaded dependencies and SDK installations are not committed.

For source imports outside Polygeist, `POLYGEIST_DEPENDENCY_ROOT` points to this
directory. Reference hashes are checked against the library manifests.
