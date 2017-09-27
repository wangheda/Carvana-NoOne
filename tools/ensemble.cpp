#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;

int main(int argc, char* argv[]) {

  cerr << "ensemble program" << endl;

  if (argc > 1) {
    int C = argc - 1;

    cerr << "open file" << endl;
    // open file
    ifstream* files = new ifstream[C];
    for (int i = 0; i < C; i++) {
      files[i].open(argv[i + 1], ifstream::in);
    }

    cerr << "create mask" << endl;
    int L = 2455040;
    float *mask = new float[L];
    float b = 1.0 / C;

    cerr << "begin merging" << endl;
    while (files[0].good()) {
      for (int j = L-1; j >= 0; --j) {
        mask[j] = 0.0;
      }

      string image_id;

      // read one line from each file
      for (int i = 0; i < C; i++) {
        ifstream& f = files[i];
        string line;
        getline(f, line);
        if (line.size() > 10) {
          stringstream ss(line);

          string im_id;
          getline(ss, im_id, ',');
          image_id = im_id;

          int start;
          int length;
          while (ss >> start >> length) {
            --start;
            for (int j = start + length - 1; j >= start; --j) {
              mask[j] += b;
            }
          }
        }
      }

      int last_zero = -1;
      cout << image_id << ',';
      for (int j = 0; j < L; ++j) {
        if (mask[j] < 0.5) {
          int first_one = last_zero + 1;
          if (j > first_one) {
            cout << first_one + 1 << ' ' << j - first_one << ' ';
          }
          last_zero = j;
        }
      }
      cout << endl;
    }

    // close file
    for (int i = 0; i < C; i++) {
      files[i].close();
    }

    delete[] mask;
    delete[] files;
  } else {
    cerr << "Usage: ./ensemble [file1] [file2] [file3] > [ensemble_file]" << endl;
  }

  return 0;
}
