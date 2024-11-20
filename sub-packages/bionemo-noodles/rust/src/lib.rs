use std::fs::File;
use pyo3::prelude::*;
use memmap2::Mmap;
use std::io;
use noodles_fasta::{self as fasta, fai};
use noodles_fasta::fai::Record;
use noodles_core::region::Region;
use std::path::{Path};

// Expose the Record struct so we can package it nicely in Python.
#[pyclass]
#[derive(Clone)]
struct PyRecord {
    name: String,
    length: u64,
    offset: u64,
    line_bases: u64,
    line_width: u64,
}

#[pymethods]
impl PyRecord {
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn length(&self) -> u64 {
        self.length
    }

    #[getter]
    fn offset(&self) -> u64 {
        self.offset
    }

    #[getter]
    fn line_bases(&self) -> u64 {
        self.line_bases
    }

    #[getter]
    fn line_width(&self) -> u64 {
        self.line_width
    }
    fn __str__(&self) -> String {
        format!(
            "PyRecord(name={}, length={}, offset={}, line_bases={}, line_width={})",
            self.name, self.length, self.offset, self.line_bases, self.line_width
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "<PyRecord name='{}' length={} offset={} line_bases={} line_width={}>",
            self.name, self.length, self.offset, self.line_bases, self.line_width
        )
    }
}

impl From<&fai::Record> for PyRecord {
    fn from(record: &fai::Record) -> Self {
        Self {
            name: String::from_utf8_lossy(record.name()).to_string(),
            length: record.length(),
            offset: record.offset(),
            line_bases: record.line_bases(),
            line_width: record.line_width(),
        }
    }
}

#[pyclass]
struct _IndexedFastaReader {
    reader: fasta::io::IndexedReader<fasta::io::BufReader<File>>,
}

#[pymethods]
impl _IndexedFastaReader {
    #[new]
    fn new(fasta_path: &str) -> PyResult<Self> {
        let fai_path = fasta_path.to_string() + ".fai";
        let fai_path = Path::new(&fai_path);  // Convert back to a Path


        let fasta_path = Path::new(fasta_path);
        // let fai_path = fasta_path.with_extension("fai");

        // Check if the .fai index file exists; if not, create it.
        if !fai_path.exists() {
            // Generate the index by reading the FASTA file
            let index = fasta::io::index(fasta_path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to create index: {}", e)))?;

            // Write the index to the .fai file
            let fai_file = File::create(&fai_path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to create .fai file: {}", e)))?;

            let mut writer = fai::Writer::new(fai_file);
            writer.write_index(&index)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to write .fai index: {}", e)))?;
        }


        let reader = fasta::io::indexed_reader::Builder::default()
            .build_from_path(fasta_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to create indexed reader: {}", e)))?;
        
        Ok(_IndexedFastaReader { reader })
    }

    fn query_region(&mut self, region_str: &str) -> PyResult<String> {
        let region: noodles_core::region::Region = region_str.parse()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid region: {}", e)))?;

        let query_result = self.reader.query(&region)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to query region: {}", e)))?;

        
        Ok(
            String::from_utf8_lossy(
                query_result.sequence().as_ref()
            ).to_string()
        )
    }

    fn records(&self) -> Vec<PyRecord> {
        // get all the entries in the index, useful for building bounds in python land.
        self.reader
            .index()
            .as_ref()
            .iter()
            .map(|record| PyRecord::from(record))
            .collect()
    }
    
}



fn region_length(region: &noodles_core::region::Region) -> Option<usize> {
    // We expect region to have both a start and an end, and end > start.
    // for the usecase where a user has only start or is empty, we do not expect to use this method. These return
    //  the entire sequence from start or the entire reference.
    let interval = region.interval();
    match (interval.start(), interval.end()) {
        (Some(start), Some(end)) => {
            let len = end.get() - start.get();
            if len > 0 {
                Some(len as usize)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn fai_record_end_in_bytes(record: &fai::Record) -> usize {
    // compute the end byte for this record excluding the newline and carriage return.
    // gonna need some tests here
    let num_bases_remain = record.length() % record.line_bases();
    let num_full_lines = record.length() / record.line_bases();
    let bytes_to_end = (num_full_lines * record.line_width() + num_bases_remain);
    return bytes_to_end as usize
}


fn read_sequence_mmap(index: &fai::Index, reader: &Mmap, region_str: &str) -> io::Result<Vec<u8>> {
    let region: noodles_core::region::Region = region_str.parse()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid region: {}", e)))?;
        
    // the string -> region transform happens on the region FromStr implementation, nice one rust!
    let start: u64 = index.query(&region)?; // byte offset for the start of this contig + sequence.

    // but we actually want the parameters for this guy too...
    let record = index.as_ref()
            .iter()
            .find(|record| record.name() == region.name())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("invalid reference sequence name: {}", region.name(),),
                )
            })?;

    if let Some(len) = region_length(&region){
        // Mental math, if we have the region length, we can compute the end of the record by adding the newline characters
       

        let mut result = vec![];
        let _ = read_sequence_limit(
            reader,
            start as usize,
            len,
            record.line_bases() as usize,
            record.line_width() as usize,
            fai_record_end_in_bytes(record),
            &mut result,
        );
        return Ok(result);
    } 
    else {
        // not really an IO error but whatever.
        return io::Result::Err(io::Error::new(io::ErrorKind::InvalidInput, "Invalid region"));
    }
}

#[pyclass]
struct IndexedMmapFastaReader {
    mmap_reader: memmap2::Mmap,
}

#[pymethods]
impl IndexedMmapFastaReader {
    #[new]
    fn new(fasta_path: &str) -> PyResult<Self> {
        let fai_path = fasta_path.to_string() + ".fai";
        let fai_path = Path::new(&fai_path);  // Convert back to a Path
        let fasta_path = Path::new(fasta_path);

        // Check if the .fai index file exists; if not, create it.
        if !fai_path.exists() {
            // Generate the index by reading the FASTA file
            let index = fasta::io::index(fasta_path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to create index: {}", e)))?;

            // Write the index to the .fai file
            let fai_file = File::create(&fai_path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to create .fai file: {}", e)))?;

            let mut writer = fai::Writer::new(fai_file);
            writer.write_index(&index)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to write .fai index: {}", e)))?;
        }

        // TODO this is where we load our shit
        let fd = File::open(fasta_path)?;
        let mmap_reader = unsafe { memmap2::MmapOptions::new().map(&fd) }?;
        
        Ok(IndexedMmapFastaReader { mmap_reader })
    }
}

fn read_sequence_limit(
    mmap: &Mmap,         // Memory-mapped file
    start: usize,        // Start position in the file (from the index)
    max_bases: usize,    // Maximum number of bases to read
    line_bases: usize,   // Number of bases per line (from the `.fai` index)
    line_width: usize,   // Number of bases per line (from the `.fai` index)
    record_end: usize,   // End position of the record in the file (from the index)
    buf: &mut Vec<u8>,   // Buffer to store the sequence
) -> io::Result<usize> {

    let mut read_count = 0;
    let mut position = start;
    let junk_offset = line_width - line_bases;

    while read_count < max_bases && position < mmap.len() && position < record_end {
        
        // get the end of the read, basically if we hit the end of the record or end of the file, we stop at that position.
        let line_end  = if record_end < (position + line_bases) {
           // Note this will have newlines and extra junk
           record_end - junk_offset
        } else if mmap.len() < (position + line_bases) {
           // Note this will have newlines and extra junk
           // we are at the end of the file
           mmap.len() - junk_offset
        } else{
            position + line_bases
        };

        // Get the slice for this line
        let line = &mmap[position..line_end];

        let base_start = position;
        let base_end = if max_bases - read_count < line_bases {
            // we have less than a full line remaining to read
            base_start + (max_bases - read_count)
        } else {
            // just take the full line (minus newline etc).
            base_start + line_bases
        };

        // Add the bases to the buffer
        buf.extend_from_slice(&line[base_start..base_end]);

        // Update the read count and position
        let new_bases_read = base_end - base_start;
        read_count += new_bases_read;
        position += new_bases_read;

        // Skip over the remaining part of the line (newlines, etc.)
        position += junk_offset
    }

    Ok(read_count)
}


#[pymodule]
fn noodles_fasta_wrapper(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<IndexedMmapFastaReader>()?;
    Ok(())
}

