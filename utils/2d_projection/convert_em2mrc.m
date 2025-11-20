% convert_em2mrc('/your/path/here');

function convert_em2mrc(root_path)
    % root_path: 输入的目录，比如 '/path/to/data/'

    % 记录开始时间
    start_time = datetime('now');
    
    % 递归寻找所有 .em 文件
    files = dir(fullfile(root_path, '**', '*.em'));
    total_files = numel(files);
    completed_count = 0;

    try
        for k = 1:numel(files)
            em_file = fullfile(files(k).folder, files(k).name);

            % 转换输出文件名：拓展名改为 .mrc
            [folder, name, ~] = fileparts(em_file);
            mrc_file = fullfile(folder, [name '.mrc']);

            % 执行转换
            dwrite(-dread(em_file), mrc_file);
            completed_count = completed_count + 1;

            fprintf('Converted: %s --> %s\n', em_file, mrc_file);
        end
    catch ME
        fprintf('Error occurred: %s\n', ME.message);
    end

    % 记录结束时间
    end_time = datetime('now');

    % 写入日志文件
    log_file = fullfile(root_path, 'convert_em2mrc.log');
    fid = fopen(log_file, 'w');
    if fid ~= -1
        fprintf(fid, 'Start time: %s\n', datestr(start_time));
        fprintf(fid, 'End time: %s\n', datestr(end_time));
        fprintf(fid, 'Input files: %d\n', total_files);
        fprintf(fid, 'Completed: %d\n', completed_count);
        fclose(fid);
    end
end