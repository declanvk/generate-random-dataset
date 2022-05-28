#![deny(missing_docs)]

//! The `generate-random-dataset` tool is used to create random datasets for
//! testing purposes. It will quickly generate large numbers of random binary
//! files.

use anyhow::{bail, Context, Result};
use argh::FromArgs;
use crossbeam::sync::WaitGroup;
use fastrand::Rng;
use log::{debug, info, warn};
use std::{
    fs::{self, OpenOptions},
    io::{self, BufWriter},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc,
    },
    thread,
    time::Instant,
};

#[derive(FromArgs, PartialEq, Debug)]
/// Top-level command.
struct GenerateRandomDatasetArgs {
    /// how large the entire dataset should be, measured in bytes
    #[argh(option)]
    max_dataset_size: u64,

    /// the filepath where the generated files will go
    #[argh(option)]
    output_folder: PathBuf,

    #[argh(subcommand)]
    file_size_distribution: FileSizeDistributions,
}

#[derive(FromArgs, PartialEq, Debug)]
#[argh(subcommand)]
enum FileSizeDistributions {
    SameSize(SameSize),
}

#[derive(FromArgs, PartialEq, Debug)]
/// Every file in the dataset is the same size.
///
/// The size of each file will be `max_dataset_size / file_count`.
#[argh(subcommand, name = "same-size")]
struct SameSize {
    /// the total number of files to generate.
    ///
    /// If the number of files is 0, it will throw a validation error. If the
    /// total dataset size is not evenly divisible by the number of files, it
    /// will produce a smaller dataset than the max size.
    #[argh(option)]
    file_count: u32,
}

fn main() -> Result<()> {
    env_logger::init();

    let abort_signal = Arc::new(AtomicBool::new(false));

    ctrlc::set_handler({
        let abort_signal = Arc::clone(&abort_signal);
        move || {
            warn!("Execution interrupted! Telling all threads to halt.");
            abort_signal.store(true, Ordering::Relaxed);
        }
    })
    .expect("Error setting ctrl-c handler");

    let args: GenerateRandomDatasetArgs = argh::from_env();

    validate_args(&args)?;

    debug!("Validated arguments: {:?}", args);

    let num_threads = thread::available_parallelism().with_context(|| {
        "Unable to retrieve available parallelism to determine number of threads to use."
    })?;

    debug!("Number of threads to use: {}", num_threads);

    let start_time = Instant::now();

    let common_args = CommonArgs {
        output_folder: &args.output_folder,
        max_dataset_size: args.max_dataset_size,
        abort_signal: Arc::clone(&abort_signal),
        num_threads: num_threads.get(),
    };

    let result = match args.file_size_distribution {
        FileSizeDistributions::SameSize(strategy) => strategy.generate_files(common_args),
    };

    let end_time = Instant::now();

    let run_duration = end_time - start_time;

    if result.is_ok() {
        info!(
            "Generating the dataset completed in [{}] seconds.",
            run_duration.as_secs_f64()
        );
    }

    result
}

fn validate_args(args: &GenerateRandomDatasetArgs) -> Result<()> {
    debug!("Validating arguments: {:?}", args);

    if args.max_dataset_size == 0 {
        bail!(
            "The max dataset size was given as 0, please increase this argument to a non-zero \
             value."
        );
    }

    if !args.output_folder.exists() {
        warn!(
            "The output folder given [{}] does not exist! Creating it now...",
            args.output_folder.display()
        );
        fs::create_dir_all(&args.output_folder).with_context(|| {
            format!(
                "Unable to create output folder at path [{}].",
                args.output_folder.display()
            )
        })?;
    }

    if !args.output_folder.is_dir() {
        bail!(
            "The output folder given [{}] is not a folder!",
            args.output_folder.display()
        );
    }

    match args.file_size_distribution {
        FileSizeDistributions::SameSize(SameSize { file_count }) => {
            if file_count == 0 {
                bail!(
                    "The number of files in the requested dataset was 0, please enter a number \
                     greater than 0."
                );
            }

            let file_size_exact = args.max_dataset_size as f64 / file_count as f64;
            if file_size_exact < 1.0 {
                bail!(
                    "For the dataset size [{}] and the number of files requested [{}], the \
                     resulting file size would be less than one byte. Please either increase the \
                     max dataset size or decrease the number of files.",
                    args.max_dataset_size,
                    file_count
                );
            }

            let file_size_rem = args.max_dataset_size % file_count as u64;
            if file_size_rem != 0 {
                warn!(
                    "The max dataset size is not perfectly divisible by the file count, with a \
                     remainder of [{}] bytes. This will result in a datset size that is less than \
                     the max requested.",
                    file_size_rem
                );
            }
        },
    }

    Ok(())
}

struct CommonArgs<'p> {
    output_folder: &'p Path,
    max_dataset_size: u64,
    num_threads: usize,
    abort_signal: Arc<AtomicBool>,
}

/// This trait covers all different strategies used to generate the files of the
/// dataset.
trait GenerateFileStrategy {
    fn generate_files<'p>(&self, args: CommonArgs<'p>) -> Result<()>;
}

impl GenerateFileStrategy for SameSize {
    fn generate_files<'p>(&self, args: CommonArgs<'p>) -> Result<()> {
        let file_count = self.file_count;
        let file_size = args.max_dataset_size / u64::from(file_count);

        info!(
            "Going to generate a dataset of approximately size [{}] into the folder [{}] with \
             [{}] total files, each of size [{}] bytes.",
            args.max_dataset_size,
            args.output_folder.display(),
            file_count,
            file_size
        );

        let file_name_id = Arc::new(AtomicU32::new(0));

        crossbeam::scope(|scope| {
            let mut threads = Vec::with_capacity(args.num_threads);
            let wait_group = WaitGroup::new();

            debug!(
                "Spawning [{}] threads to generate [{}] files...",
                args.num_threads, file_count
            );
            for _ in 0..args.num_threads {
                threads.push(scope.spawn({
                    let wg = wait_group.clone();
                    let args = &args;
                    let file_name_id = Arc::clone(&file_name_id);
                    move |_scope| {
                        let result = same_size_thread(file_name_id, file_size, file_count, args);
                        drop(wg);
                        result
                    }
                }));
            }

            debug!("Waiting for worker threads to finish...");
            wait_group.wait();

            threads
                .into_iter()
                .map(|t| t.join().expect("unable to join thread"))
                .collect::<Result<()>>()
        })
        .expect("unable to extract scope return type, some threads must have failed")?;

        Ok(())
    }
}

fn same_size_thread(
    file_name_id: Arc<AtomicU32>,
    file_size: u64,
    file_count: u32,
    args: &CommonArgs<'_>,
) -> Result<()> {
    const BUFFER_SIZE: u64 = 4096;
    let mut output_buffer = [0; BUFFER_SIZE as usize];
    let rng = Rng::new();

    loop {
        test_abort_signal(&args.abort_signal)?;

        let next_filename = file_name_id.fetch_add(1, Ordering::SeqCst);

        if next_filename >= file_count {
            // We've generated all the files
            return Ok(());
        }

        debug!("Starting generating file with id [{}].", next_filename,);

        let file_path = args
            .output_folder
            .join(PathBuf::from(format!("{}.bin", next_filename)));
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&file_path)
            .with_context(|| format!("Failed to open [{}] for writing.", file_path.display()))?;
        let mut writer = BufWriter::new(file);

        let mut num_bytes_written = 0;

        while num_bytes_written < file_size {
            let num_bytes_to_fill =
                usize::try_from(std::cmp::min(BUFFER_SIZE, file_size - num_bytes_written)).unwrap();
            let buffer_slice = &mut output_buffer[..num_bytes_to_fill];
            fill_buffer(&rng, buffer_slice);

            num_bytes_written += io::copy(&mut buffer_slice.as_ref(), &mut writer)
                .with_context(|| "Unable to copy bytes from random buffer into file.")?;
        }

        debug_assert_eq!(
            num_bytes_written, file_size,
            "either overfilled or underfilled the file, should be exact"
        );

        debug!(
            "Finished generating file [{}] with size [{}].",
            file_path.display(),
            num_bytes_written
        );
    }
}

fn fill_buffer(rng: &Rng, buffer: &mut [u8]) {
    let buffer_size = buffer.len();
    let num_windows = buffer_size / 8;

    for window_idx in 0..num_windows {
        let value_bytes = rng.u64(..).to_ne_bytes();
        let buffer_idx = window_idx * 8;

        buffer[buffer_idx..(buffer_idx + 8)].copy_from_slice(&value_bytes);
    }

    for buffer_idx in (num_windows * 8)..buffer_size {
        buffer[buffer_idx] = rng.u8(..);
    }
}

fn test_abort_signal(signal: &AtomicBool) -> Result<()> {
    if signal.load(Ordering::Relaxed) {
        bail!("Execution interrupted! Returning...")
    }

    Ok(())
}
