def run(
    user_id: str,
    forward_port: int,
    cpus: float,
    memory: int,
    gpus: List[int] or str,
    image: str or None,
    exec_command: str or None,
    ram_size: int,
    volumes_ls: List[List[str]],
):
    '''
    `user_id`: student ID.\n
    `forward_port`: which forward port you want to connect to port: 2(SSH).\n
    `cpus`: Number of CPU utilities.\n
    `memory`: Number of memory utilities.\n
    `gpus`: List of gpu id used for the container.\n
    `image`: Which image you want to use, new std_id will use "rober5566a/aivc-server:latest"\n
    `exec_command`: The exec command you want to execute when the docker runs.\n
    `ram_size`: The DRAM size that you want to assign to this container,\n
    `volumes_ls`: List of volume information, format: [[host, container, ]...]
    '''

    volume_info = ' -v '.join(':'.join(volume_ls) for volume_ls in volumes_ls)


run()