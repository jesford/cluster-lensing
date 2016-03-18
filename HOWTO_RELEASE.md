# How to Release

Here's my step-by-step guide for cutting a new release of
**cluster-lensing**. Thanks to Jake VanderPlas for showing the way!

## Pre-release

1. update version in ``setup.py``

2. update version in ``clusterlensing/__init__.py``

3. update version in ``docs/conf.py`` (two places!)

4. update change log in ``CHANGES.md``

5. update ``README.md`` and `` demo.ipynb`` if necessary

6. confirm all tests pass & flake8 gives no errors
  ```
  $ cd clusterlensing/
  $ nosetests
  $ flake8 --ignore N802,N806 `find . -name \*.py | grep -v setup.py | grep -v /doc/`
  ```

7. create a release tag; e.g.
   ```
   $ git tag -a v0.1.1 -m 'version 0.1.1 release'
   ```

8. push the commits and tag to github; e.g.

  ```
  $ git push origin v0.1.1
  $ git push origin master
  ```

9. confirm that CI tests pass on github

10. under "tags" on github, update the release notes

11. push the new release to PyPI:
   ```
   $ python setup.py sdist upload
   ```

12. change directories to ``docs`` and build the documentation:
   ```
   $ cd docs/
   $ make html       # build documentation
   $ make publish  # publish to github pages
   $ make publish  # and again to really publish
   ```

## Post-release

1. update version in ``clusterlensing/__init__.py`` to next version; e.g. '0.3-git'

2. update version in ``docs/conf.py`` to the same (in two places)

3. other version updates?
