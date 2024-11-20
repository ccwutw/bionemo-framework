# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from noodles_fasta_wrapper import IndexedFastaReader


def main():
    # Path to the sample FASTA file
    fasta_path = "test.fa"
    fasta_path = "sample.fasta"

    # Initialize the indexed reader
    reader = IndexedFastaReader(fasta_path)
    # actually expects the .fai file to exist... wtf?
    print("Indexed reader created successfully.")

    # Test querying a region
    region_str = "chr1:1-12"  # Adjust based on your requirements
    sequence = reader.query_region(region_str)
    print(f"Sequence for {region_str}: {sequence}")


    # Test failure of query
    try:
        # NOTE: it does not fail on out of bounds for the query, just returns emtpy string/truncated
        #           if seqid is invalid or the region is invalid it will fail though.
        region_str = "chr11:1-2"
        sequence = reader.query_region(region_str)
        print(sequence)
    except Exception as e:
        print(e)


    print(reader.records())

if __name__ == "__main__":
    main()
