# Publishing guide (GitHub + Zenodo)

> Maintainer notes. This file documents how the repository is published and archived.
> It is safe to keep in the repository, or to delete once the first release is out.

The workflow is: **develop on GitHub → connect Zenodo → publish a release → obtain a citable DOI**. Both services are free.

---

## Part 0 · Before uploading: replace the placeholders

The GitHub account has already been set throughout the repository (`arianadelg`), so the
URLs in `README.md`, `Neuron_ASD.ipynb` and `CITATION.cff` are ready to use. If the
repository is ever moved to a different account or organization, search for
`arianadelg/neuron-asd` and update it in those three files.

One placeholder remains, and it is filled in at the end of Part 2:

| Find | Replace with | Appears in |
|---|---|---|
| `zenodo.XXXXXXX` | the DOI Zenodo issues (Step 2.5) | `README.md` (badge and citation) |

**Optional but recommended:** add ORCID identifiers for the authors in `CITATION.cff`. They are commented out with `#`; remove the `#` and insert the identifier. ORCIDs can be created free of charge at [orcid.org](https://orcid.org).

---

## Part 1 · Create the GitHub repository

### 1.1 Create an empty repository

1. Sign in at [github.com](https://github.com).
2. Top right, click **+** → **New repository**.
3. Fill in:
   - **Repository name:** `neuron-asd`
   - **Description:** `Open platform for exploring receptor modulations toward a typically-developing EEG profile in autism`
   - **Public** — required, because Zenodo only archives public repositories
   - Do **not** tick "Add a README file" (one is already included)
   - Do **not** add a .gitignore or a license (both are already included)
4. Click **Create repository**.

### 1.2 Upload the files

**Option A — from the browser (simplest, nothing to install):**

1. On the new repository page, click **uploading an existing file**.
2. Unzip `neuron-asd-repo.zip` on your computer.
3. Drag **the contents** (the files and folders inside, not the containing folder) into the browser window.
4. Under *Commit changes*, write: `Initial release of Neuron ASD v1.0.0`
5. Click **Commit changes**.

> Files beginning with a dot (`.gitignore`, `.zenodo.json`) are hidden by default in most file explorers. On Windows enable *View → Hidden items*; on macOS press `Cmd + Shift + .` Make sure they are uploaded too.

**Option B — from the command line (if git is installed):**

```bash
cd folder-where-you-unzipped
git init
git add .
git commit -m "Initial release of Neuron ASD v1.0.0"
git branch -M main
git remote add origin https://github.com/arianadelg/neuron-asd.git
git push -u origin main
```

### 1.3 Check the result

The repository page should show:

- the rendered `README.md` with its badges;
- a **"Cite this repository"** box in the right-hand sidebar, generated automatically from `CITATION.cff` (if it is missing, check the file name is exactly `CITATION.cff`);
- the `neuron_asd/` folder containing `app.py`, `engine.py` and `__init__.py`;
- the `examples/` folder containing the two `.zip` bundles.

### 1.4 Test the notebook in Colab

Click the **"Open In Colab"** badge in the README. The notebook should open. Run **Step 1**, then **Step 5** (the built-in example) to confirm that installation from GitHub works. If Step 1 fails, check the repository name and account in the installation URL.

---

## Part 2 · Connect Zenodo and obtain the DOI

### 2.1 Create a Zenodo account

1. Go to [zenodo.org](https://zenodo.org).
2. Click **Sign up** → choose **Sign up with GitHub**. This creates the account and links both services in one step.
3. Authorize access when GitHub asks.

### 2.2 Enable the repository in Zenodo

1. In Zenodo, open your account menu (top right) → **GitHub**
   (direct URL: `https://zenodo.org/account/settings/github/`).
2. Find `neuron-asd` in the list of repositories.
3. Switch the toggle on the right to **ON**.

> If the repository is not listed, click **Sync now**. Remember it must be public.

**Important:** Zenodo archives from this point onward. It will only capture releases created **after** the toggle is switched on.

### 2.3 Create the release on GitHub

1. Return to the repository on GitHub.
2. In the right-hand sidebar, **Releases** → **Create a new release**.
3. Fill in:
   - **Choose a tag:** type `v1.0.0` and select *"Create new tag: v1.0.0 on publish"*
   - **Release title:** `Neuron ASD v1.0.0`
   - **Describe this release:** for example
     > First public release accompanying the paper. Includes the modelling engine, the user interface, the Colab notebook, the user guide, and synthetic example data.
4. Click **Publish release**.

### 2.4 Zenodo archives it automatically

Within a few minutes Zenodo detects the release, archives it and issues the DOI. To check:

1. Go back to `https://zenodo.org/account/settings/github/`.
2. A **DOI badge** appears next to `neuron-asd`.
3. Open the record and confirm the metadata (title, authors, license) is correct — it comes from `.zenodo.json`. Anything wrong can be corrected in Zenodo with **Edit**.

**About the two DOIs:** Zenodo issues a *concept DOI* (always resolves to the most recent version) and a *version DOI* (resolves to v1.0.0 specifically). Cite the **concept DOI** in the paper, so the citation does not go stale when new versions are released.

### 2.5 Add the DOI to the README

1. On the Zenodo record page, copy the **Markdown** badge snippet.
2. On GitHub, edit `README.md` (pencil icon) and replace both occurrences of `zenodo.XXXXXXX`:
   - the badge line at the top;
   - the citation line in the **Citation** section.
3. Commit the change.

The software now has a citable DOI.

---

## Part 3 · Link it to the article

The data/code availability statement in the manuscript should point to the DOI rather than to GitHub alone: a repository can move or disappear, a DOI cannot. Suggested wording:

> The Neuron ASD platform is openly available at https://github.com/arianadelg/neuron-asd and archived at Zenodo (DOI: 10.5281/zenodo.XXXXXXX). The user guide and synthetic example data are included in the repository.

If the journal also asks about the evaluation data, note that all three EEG datasets are already public (Sheffield ORDA; OpenNeuro ds003775; OpenNeuro ds005385) and are cited in the article.

---

## Part 4 · Future versions

When something is fixed or added:

1. Push the changes to the repository.
2. Update the version number in `setup.py`, `neuron_asd/__init__.py` and `CITATION.cff`.
3. Create a **new release** on GitHub (for example `v1.0.1`).
4. Zenodo archives it automatically and issues a new *version DOI*, while the *concept DOI* stays the same.

The setup does not need to be repeated; the Zenodo toggle remains enabled.

---

## Troubleshooting

**The repository does not appear in the Zenodo list.**
It must be public, and you may need to click **Sync now**. If it still does not appear, check under GitHub → *Settings → Applications → Authorized OAuth Apps* that Zenodo has permission.

**The release was published but Zenodo did not archive it.**
Almost always because the toggle was switched on *after* the release was created. Fix: create a new release (for example `v1.0.1`) with the toggle already enabled.

**The "Cite this repository" box does not appear on GitHub.**
The file must be named exactly `CITATION.cff`, sit in the repository root, and be valid YAML. If it was edited by hand, check that indentation still uses spaces rather than tabs.

**The Colab notebook fails at Step 1.**
Usually the repository is still private, or the account/repository name in the installation URL does not match. Open the notebook on GitHub and check the two lines containing `github.com/arianadelg/neuron-asd`.
