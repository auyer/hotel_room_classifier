package main

import (
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path"
	"path/filepath"
	"sync"
)

var DEST = "newtrain"
var ORIG = "train"
var fnameSeq = 0
var wg sync.WaitGroup

// File copies a single file from src to dst
func File(src, dst string) error {
	var err error
	var srcfd *os.File
	var dstfd *os.File
	var srcinfo os.FileInfo
	if _, err = os.Stat("/path/to/whatever"); os.IsNotExist(err) {
		if srcfd, err = os.Open(src); err != nil {
			return err
		}
		defer srcfd.Close()

		if dstfd, err = os.Create(dst); err != nil {
			return err
		}
		defer dstfd.Close()

		if _, err = io.Copy(dstfd, srcfd); err != nil {
			return err
		}
		if srcinfo, err = os.Stat(src); err != nil {
			return err
		}
		return os.Chmod(dst, srcinfo.Mode())
	}
	return nil

}

func Dir(src string, dst string) error {
	var err error
	var fds []os.FileInfo

	if fds, err = ioutil.ReadDir(src); err != nil {
		return err
	}
	for _, fd := range fds {
		srcfp := path.Join(src, fd.Name())
		dstfp := path.Join(dst, fd.Name())

		if fd.IsDir() {
			wg.Add(1)
			fmt.Println(fmt.Sprintf("Starting %s path", src))
			go func(srcfp, dstfp string) {
				if err = Dir(srcfp, dstfp); err != nil {
					fmt.Println(err)
				}
				wg.Done()
				fmt.Println(fmt.Sprintf("Done %s path", src))
			}(srcfp, dstfp)

		} else {
			fnameSeq++
			if err = File(srcfp, path.Join(DEST, fmt.Sprintf("%07d", fnameSeq)+filepath.Ext(fd.Name()))); err != nil {
				fmt.Println(err)
			}
		}
	}
	return nil
}

func main() {
	if len(os.Args[1:]) == 2 {
		ORIG = os.Args[1]
		DEST = os.Args[2]
	}

	if err := os.MkdirAll(DEST, os.ModePerm); err != nil {
		log.Fatal(err)
	}
	fmt.Println("Starting")
	Dir(ORIG, DEST)
	wg.Wait()
}
