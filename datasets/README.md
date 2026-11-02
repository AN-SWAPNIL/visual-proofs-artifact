# Dataset Access and Verification

## Conditions

`r` is the randomized collection. `wr` is the collection without randomization.
Both contain 24,000 images, not 24,000 images combined. Within each collection
there are eight properties, three categories, and 1,000 images per group.

The randomized archive also contains `reasoning_texts.json`. Both contain
`property_labels.json`. Images are organized as:

```text
graphtrails_dataset/
  cut_vertex/{yes,s_no,d_no}/*.png
  connectivity/{yes,s_no,d_no}/*.png
  bipartite/{yes,s_no,d_no}/*.png
  planarity/{yes,s_no,d_no}/*.png
  outer_planarity/{yes,s_no,d_no}/*.png
  hamiltonian/{yes,s_no,d_no}/*.png
  k_colorability/{yes,s_no,d_no}/*.png
  completeness/{yes,s_no,d_no}/*.png
  property_labels.json
  reasoning_texts.json       # r only
```

`manifest.json` records archive sizes, SHA-256 checksums, category counts,
Kaggle slugs, and whether a reasoning file exists. `r_files.csv` and
`wr_files.csv` record the byte size and SHA-256 of every image/metadata file.
These identify the exact local release contents independently of ZIP encoding.
Neither filenames nor hashes establish graph-isomorphism independence.

The actual archived image dimensions and metadata ranges are:

| Collection | PNG dimensions | Node counts reported in metadata |
| --- | --- | --- |
| `r` | 400x400 for all 24,000 images | 10--60 |
| `wr` | 406x406 for all 24,000 images | 14--40 |

The non-randomized renderer uses tight bounding-box output, so its actual
dimensions differ from the nominal 400x400 canvas. The manifest records
observed dimensions, not only the generator's stated nominal resolution.

## Obtain the Data

The initial prepared local repository already has both archives in
`datasets/archives/`. This directory is not tracked by Git. For a fresh clone,
download the relevant dataset from its Kaggle page:

- [Randomized](https://www.kaggle.com/datasets/ahmmadnurswapnil/graph-visualization)
- [Without randomization](https://www.kaggle.com/datasets/ahmmadnurswapnil/graph-visualization-wr)

Alternatively, with the official Kaggle CLI installed and authenticated:

```sh
kaggle datasets download -d ahmmadnurswapnil/graph-visualization -p datasets/archives
kaggle datasets download -d ahmmadnurswapnil/graph-visualization-wr -p datasets/archives
```

Do not commit credentials, `kaggle.json`, or downloaded archives. These
addresses are inferred from historical notebook mounts; live availability,
version numbers, and license metadata still require author confirmation.

## Verify and Extract

From the repository root:

```sh
python scripts/datasets.py list
python scripts/datasets.py verify --condition r
python scripts/datasets.py verify --condition wr
python scripts/datasets.py extract --condition r
python scripts/datasets.py extract --condition wr
```

Extraction creates `datasets/extracted/r/graphtrails_dataset` and
`datasets/extracted/wr/graphtrails_dataset`. Existing nonempty directories are
not overwritten. The helper verifies contents before extracting and rejects
path traversal and symbolic links in an archive.

Kaggle's download ZIP may have a different name, compression, or wrapper
directory from the local reference archive. Supply its actual path:

```sh
python scripts/datasets.py extract --condition r --archive datasets/archives/graph-visualization.zip
python scripts/datasets.py extract --condition wr --archive datasets/archives/graph-visualization-wr.zip
```

A different ZIP checksum is acceptable only when every uncompressed file
matches the release's size and SHA-256 manifest. A content mismatch is an
error, not an instruction to silently replace the reference manifest.

## Hosting and Licensing

Keep the full data on Kaggle and, after approval, optionally attach the two
archives as GitHub Release assets. GitHub blocks individual files larger
than 100 MiB in regular Git, and these archives are approximately 956 MB and
570 MB. See [GitHub's large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
Do not use Git LFS merely to duplicate data already hosted on Kaggle unless
the authors deliberately choose to manage its storage and bandwidth.

The proposed dataset release uses [CC BY 4.0](LICENSE.md). Confirm author
approval and the existing Kaggle licenses before publishing. Code uses MIT;
the dataset license does not apply to third-party model weights or the paper.

