package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"
)

var NumClass = 365
var Workers = 18
var CategoriesFile = "categories_places365.txt"
var TrainFile = "places365_train_challenge.txt"
var ValFile = "places365_val.txt"

var categoriesMap map[int]string
var wg sync.WaitGroup

func worker(id int, linePrefix, dirPrefix string, linesChan <-chan string) {
	for line := range linesChan {
		processFileAndCopy(linePrefix+line, dirPrefix)
	}
	fmt.Println("Finished worker", id)
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func lineCaller(fileLocation string, linesChan chan string) {
	file, err := os.Open(fileLocation)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		linesChan <- scanner.Text()
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
	wg.Done()
}

// copyFile copies a single file from src to dst
func copyFile(src, dst string) error {
	var err error
	var srcfd *os.File
	var dstfd *os.File
	var srcinfo os.FileInfo

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

func separateAndConvert(line string) (int, string) {
	lineParts := strings.Split(line, " ")
	classID, err := strconv.Atoi(lineParts[1])
	check(err)
	return classID, lineParts[0]

}

func dirCreator(class string) {
	os.MkdirAll("./places2_hotels/train/"+class, os.ModePerm)
	os.MkdirAll("./places2_hotels/val/"+class, os.ModePerm)
}

func closeChanWhenDone(linesChan chan string) {
	wg.Wait()
	close(linesChan)
}

func processFileAndCopy(line, prefix string) {
	classID, dir := separateAndConvert(line)
	filename := strings.Split(dir, "/")
	copyFile("./"+prefix+dir, "./places2_hotels/"+prefix+"/"+categoriesMap[classID]+"/"+filename[len(filename)-1])
}

func main() {
	linesChan := make(chan string, 50)
	wg.Add(1)
	go lineCaller(CategoriesFile, linesChan)

	categoriesMap = make(map[int]string)

	var line string
	for i := 0; i < NumClass; i++ {
		line = <-linesChan
		classID, classPath := separateAndConvert(line)
		className := strings.Split(classPath, "/")
		categoriesMap[classID] = className[len(className)-1]
		go dirCreator(className[len(className)-1])
	}

	wg.Wait()
	fmt.Println("Created Directories")
	wg.Add(1)
	fmt.Println("Moving Train Files")
	go lineCaller(, linesChan)
	for w := 1; w <= Workers; w++ {
		go worker(w, TrainFile, "train", linesChan)
	}
	closeChanWhenDone(linesChan)
	fmt.Println("Copied all Train Files")

	wg.Add(1)
	linesChan = make(chan string, 50)
	fmt.Println("Moving Validation Files")
	go lineCaller(ValFile, linesChan)
	for w := 1; w <= Workers; w++ {
		go worker(w, "/", "val", linesChan)
	}
	closeChanWhenDone(linesChan)

	fmt.Println("Copied all Validation Files")
}
