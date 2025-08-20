# Copy datas from local to g5k

We can use `scp` or `rsync`. 

- With `rsync`: 

```bash
rsync -avzPL ~/projects/spartDM/data/starmen bdang@grenoble.g5k:~/project/data

rsync -avzP --exclude '/data/' --exclude '.venv/' --exclude '/archive/' ~/projects/spartDM/ bdang@grenoble.g5k:~/project/spartDM

```

Flags of `rsync`: 

- `-a`: Archive mode: preserves permissions, symlinks, timestamps, groups, etc. Equivalent to -rlptgoD. It's a common choice when copying full directories.

- `-v`: Verbose: shows detailed output about what rsync is doing.

- `-z`:	Compress: compresses file data during transfer. Useful over networks to save bandwidth.

- `-P`:	A shortcut for two flags:
	    --partial: keep partially transferred files if the transfer is interrupted (so it can resume).
	    --progress: show progress during transfer.

- `--delete`: mirror exactly the source and destination. If files are deleted in source -> delete these in destination. 

- With `scp`: 

```bash
scp source remotelogin@rhost:/tmp
```

**Important**: With scp, `scp source rlogin@rhost:/tmp` and `scp source/ rlogin@rhost:/tmp` are the same. With rsync, they aren't :

- With source, a directory named source is created in /tmp, with the files in it.

- With source/, the files in source are copied directly in /tmp.


## Create symlink in cluster

```bash
ln -s ~/project/data/starmen ~/project/spartDM/data/
```

-Note: this will create a new folder `starmen` in the project/data directory. 


## Sync workdir from cluster to local to get the result

```bash
rsync -avxP  bdang@grenoble.g5k:~/project/spartDM/{workdir,oarlogs} ~/projects/spartDM/
```


# Connect to cluster (from frontend machine)

Grom grenoble, the best cluster is `kinovis` with 6 nodes - each node has 2 x L40S GPUs. 

```bash
oarsub -q besteffort -l "nodes=1/gpu=1" -I

oarsub -q besteffort -p cluster='kinovis' -l "nodes=1/gpu=2" -I
```

- `p`: property - query by SQL syntax. It is best practice to enclose query in double quote `""`.

## Connect to cluster with specific private key (for debug in VSCode)

You must add `-i ~/.ssh/id_rsa_g5k_internal` to your oarsub reservation commands:

```bash
oarsub -i ~/.ssh/id_rsa_g5k_internal -q besteffort -p cluster='kinovis' -l "nodes=1/gpu=1" -I
```

Then connect from VSCode using ssh `*.g5koar` with `*` is the name of host, e.g: vercors2-2

To check host, using several methods: 

```bash
oarsub -C <job_id>

hostname
```

## Nancy cluster - P3 priority



## Job status - inside node

- Check job id: 

```bash
oarstat -u login
```

- Check job state: 
```
oarstat -j <job_id>
```

- Kill a job
```
oardel <job_id> 
```

## Check gpu infos inside a node

```bash
nvidia-smi -l 2
nvidia-smi topo --matrix

```