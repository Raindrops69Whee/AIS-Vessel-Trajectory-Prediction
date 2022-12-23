import pandas as pd
import pickle as pkl
import sys

class Processor:

    def __init__(self):
        pass

    def process(self, filenames, multiple_files=False, verbose=True, logfile=None, req_fields=None):
        if logfile is not None:
            self.logfile=logfile
        else:
            self.logfile="./templog.txt"
        out=sys.stdout
        if req_fields is not None:
            self.req_fields=req_fields
        else:
            self.req_fields=[]
        if not verbose:
            sys.stdout=open(self.logfile)
        error_files=[]
        datas={}
        if multiple_files:
            self.filenames=[i for i in filenames]
        else:
            self.filenames=[]
            self.filenames.append(filenames)
        for filename in self.filenames:
            try:
                text=""
                with open(filename, "r") as f:
                    text = f.read()
                text = text[1:-1].split(",\n")
                data = []
                for i in range(len(text)):
                    text[i] = text[i].replace("null", "None")
                    data.append(eval(text[i]))
                remove_keys=[]
                for i in range(len(data)):
                    for j in self.req_fields:
                        if j not in data[i].keys():
                            remove_keys.append(i)
                for i in remove_keys[::-1]:
                    data.pop(i)
                datas[filename]=data
            except Exception:
                print("Error occurred! The file that caused this error was:", filename)
                error_files.append(filename)
        if multiple_files:
            if len(error_files)==len(filenames):
                print("None of the files were processed successfully!")
                sys.stdout=out
                return None
            print("Number of files successfully processed:", len(datas))
            print("Number of error files:", len(error_files))
            if len(error_files)>0:
                print("List of files that were unable to be processed:")
                for name in error_files:
                    print(name)
            sys.stdout=out
            return datas
        else:
            if len(error_files)>0:
                print("File was not successfully processed.")
                out=sys.stdout
                return None
            print("File was successfully processed!")
            sys.stdout=out
            return datas[filenames]
    def to_pickle(self, datas, filename):
        try:
            with open(filename, 'wb') as f:
                pkl.dump(datas, f)
            print("File successfully 'pickled'.")
        except Exception:
            print("Error occurred!")
            return False
        return True
    def json_to_pkl(self, filenames, outfile, multiple_files=False, verbose=True, logfile=None, req_fields=None):
        result=self.to_pickle(self.process(filenames, multiple_files, verbose, logfile, req_fields), outfile)
        if result:
            print("Conversion successful!")
        else:
            print("Conversion unsuccessful!")
        return result