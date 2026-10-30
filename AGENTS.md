# Agent Guidelines

## Keep every tracked file at or below 50 MB

GitHub rejects individual files above 100 MB and warns above 50 MB. Treat
50 MB as the hard limit for anything committed to this repository.

- Split large binary artifacts (wheels, models, archives) into volumes of at
  most 50 MB before committing:

  ```sh
  split -b 45M -d -a 2 --numeric-suffixes=0 artifact.whl artifact.whl.part
  sha256sum artifact.whl > artifact.whl.sha256
  rm artifact.whl
  ```

- Commit the `*.partNN` volumes and the `*.sha256` file. Never commit the
  combined artifact, and make sure `.gitignore` does not re-include it.
- Reassemble and verify the artifact inside the Dockerfile or build script,
  then delete the temporary combined file:

  ```dockerfile
  COPY path/to/artifact.whl.part* path/to/artifact.whl.sha256 /app/wheels/
  RUN cd /app/wheels && \
      cat artifact.whl.part* > artifact.whl && \
      sha256sum -c artifact.whl.sha256 && \
      ... use artifact.whl ... && \
      rm -rf /app/wheels
  ```

- See `docker/DockerfileSYI` for the sycl-infer wheel and
  `docker/DockerfileOpenVINO` for the OpenVINO wheel.
- If a file already above the limit has to be modified, split it in that
  change instead of growing it further.
