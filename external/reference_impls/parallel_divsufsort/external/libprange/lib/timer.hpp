#include <chrono>
#include <iostream>
#include <unordered_map>

using namespace std;

class Timer {
	private:
		static unordered_map<string,
			vector<chrono::time_point<chrono::steady_clock>>
				> measurements;
	public:
		static void start(string id) {
			measurements[id].push_back(chrono::steady_clock::now());
		}

		static void stop(string id) {
			measurements[id].push_back(chrono::steady_clock::now());
		}

		static void print() {
			for (auto p : measurements) {
				string id = p.first;
				auto vec = p.second;
				cout << id << " :" << endl;
				double sum = 0;
				for (size_t i = 0; i < vec.size(); i +=2) {
					auto diff = chrono::duration <double, milli>(vec[i+1] - vec[i]).count();
					cout << diff << "ms\n"; 	
					sum += diff;
				}
				cout << "Sum: " << sum << "ms" << endl << endl;
			}
		}
};
unordered_map<string,vector<chrono::time_point<chrono::steady_clock>>>
				Timer::measurements;

