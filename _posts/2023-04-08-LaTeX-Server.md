---
layout: post
title: "LaTeX on Server"
date: 2023-04-08 09:13:46 +0800
tags: [Programming]
---

> 文中提到的问题已经解决！


今天尝试在课题组服务器上使用安装好的LaTeX环境。

通过`module avail`查看可用的模块，发现有`TeXLive2021`，于是加载模块：`module load TeXLive2021`。

然后在`/home/username`下创建一个`test.tex`文件，内容如下：

```tex
\documentclass{article}
\begin{document}
First document. This is a simple example, with no 
extra parameters or packages included.
\end{document}
```

然后在命令行中输入`pdflatex test.tex`，即可生成`test.pdf`文件。

为了建立完整的工作流，接下来需要从Overleaf上下载项目，然后上传到服务器上，再编译生成PDF文件。

## 从Overleaf下载项目

在Overleaf上，点击`Project`，然后点击`Download`，选择`Download project as zip`，即可下载项目。

## 上传项目到服务器

通过vscode的`Remote-SSH`插件，可以很方便地将本地的项目上传到服务器上。在此之前先解压下载的文件。

## 编译生成PDF文件

在服务器上，进入项目文件夹，然后输入`pdflatex main.tex`，即可生成`main.pdf`文件。但返回了一个错误：

```bash
OK, entering \batchmode

kpathsea: Running mktexpk --mfmode / --bdpi 600 --mag 1+123/600 --dpi 723 uniyou7a
mktexpk: don't know how to create bitmap font for uniyou7a.
mktexpk: perhaps uniyou7a is missing from the map file.
kpathsea: Appending font creation commands to missfont.log.
```

这是因为缺少字体文件。

另外尝试了`xelatex main.tex`，也是报错：

```bash
LaTeX Warning: Reference `eq:attention' on page 5 undefined on input line 80.

! Undefined control sequence.
l.98 ...\ln{\Big[\frac{\pi(x)}{p(x)}\Big]} = \lang
                                                   \ln{\frac{\pi}{p}} \rang ...
```

这是因为缺少`amsmath`包，需要在`main.tex`文件中添加`\usepackage{amsmath}`。但main.tex中已经有`\usepackage{amsmath}`，所以应该是缺少其他包。

在Overleaf上编译没有问题，而在服务器上编译输出的pdf文件中，公式和编号都正常，但目录显示不出来。

下面是main.tex包含的包：

```tex
\usepackage[authoryear,list, math]{Style/artratex}% document settings
\usepackage{amsmath,amssymb}
\usepackage{mathrsfs} % added by Paul for special meta dataset of datasets letter D
\usepackage[dvipsnames]{xcolor}
\usepackage{multirow}
\usepackage{graphicx}
```

目录显示不出来的问题，是因为缺少`tocloft`包，需要在`main.tex`文件中添加`\usepackage{tocloft}`。

虽然在Overleaf上没有加这个包也能正常显示目录，但我仍然添加了，然后编译，目录显示了，但格式不符合要求。(标题字体太大)

为什么Overleaf上不需要这个包，而服务器上需要呢？答案可能是Overleaf上的`tocloft`包版本不同，或者Overleaf上的`tocloft`包已经包含了其他包。

另外，其实不排除我使用的编译器不同，Overleaf上使用的是`xelatex`，而服务器上使用的是`pdflatex`。

为了解决编译器不同的问题，我安装了LaTeX-in-VSCode插件，它可以给我提供许多编译器的快捷使用方式，但插件无法正常工作，原因是TexLive不在系统路径上，通过`which pdflatex`查看，发现`pdflatex`在`/usr/local/texlive/2021/bin/x86_64-linux/pdflatex`，所以需要将`/usr/local/texlive/2021/bin/x86_64-linux`添加到系统路径上。所用命令是`export PATH=$PATH:/usr/local/texlive/2021/bin/x86_64-linux`。

但没有成功使用这个插件，为了不mess up with PATH，我决定放弃这个插件。

> 实际上，我还是messed up with PATH，一度导致我的ls, cd之类的命令都无法使用，但export之类的还是可以的，这才让我意识到这些命令的级别不同（export显然更底层）， 最后我通过 `usr/bin/vim`的方式调用vim，成功修改了PATH。

从ucas-thesis的README.md中知道，如果出现了`! Undefined control sequence.`错误，应该是TexLive版本过低。但我的版本是是TeXLive2021，应该是最新的版本了。

## 回顾Overleaf上的编译

我重新检查了Overleaf，发现那边也有报错，但是编译出来的PDF文件中，公式和编号都正常，目录也显示出来了。

报错主要是Undefined control sequence，主要是\braket 和\lang \rang命令，可以替换成\left\langle \right\rangle 和\left\lvert \right\rvert。

为了避免去修改，可以在main.tex文件中添加：

```tex
\newcommand{\braket}[1]{\left\langle #1 \right\rangle}
```

就解决了这个问题。

在Overleaf上使用的编译设置是Texlive 2022, XeLaTeX。于是我修改为Texlive 2021, XeLaTeX，编译仍然成功了，说明不是编译器的问题。

---

目前暂时没有找到很好的方法解决目录消失的问题，目前的效果如图：

![空荡荡的目录](https://pic4.zhimg.com/80/v2-6cdcb951e5eb54e74b8a7f64ba6a575c.png)

正常的效果应是：

![正常效果](https://pic4.zhimg.com/80/v2-c47f139743544e8fec15781fb86df5e3.png)

不过其他方面（包括编号）均完全正常，因此初步完成了本文的目标：在服务器上使用安装好的TexLive环境。

目录的话……之后再放到Overleaf跑一次罢。

---

2023-04-09 16:11:53

## Update：问题解决

今天重新在服务器上load了插件，发现插件提供了这样一个长命令：

```bash
xelatex -no-pdf -synctex=1 -interaction=nonstopmode -file-line-error -recorder  "/home/osgood/new_thesis/Thesis.tex"
```

我果断运行了这段代码，然后发现的确`no-pdf`，于是我继续再运行了一遍`xelatex Thesis.tex`，发现得到的pdf有正常的目录！我据此认为第一步的长命令有很大的作用。

但插件仍然无法完全运行，因为会返回错误

```xelatex: /lib64/libz.so.1: version `ZLIB_1.2.9' not found (required by /cm/shared/apps/anaconda3-2021.11/lib/./libpng16.so.16)```

但这对我来说不是问题了。本文提到的困难完美解决。

接下来就是把这个编译作为脚本自动执行即可。（#TODO 参考make的使用）
