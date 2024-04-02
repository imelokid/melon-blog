---
title: 对Try-Catch-Finally一无所知
date: 2024-04-02 10:33:02
tags: [JAVA]
---
### 问题
有个常见又看似简单的问题:那就是当我们分别在JAVA的try，catch，finally返回值时，到底哪个是生效的，哪些代码被确实运行了？本文将通过JAVA字节码的方式为大家彻底解决这个问题。


### 先说结论
1. finally代码会被强行插入到try和catch代码的后面，所以无论如何，finally都会被执行。
2. 如果finally中执行了return，那么函数无论是否有异常，都会返回finally的返回值。所以，finally不要return!。
3. 由于finally代码是插到了try和catch的代码块后面，所以finally的执行逻辑要在try和catch的后面。


看如下代码
```java
public static void main(String[] args) {
        test();
    }

    public static int test() {
        try {
            System.out.println("try");
            return 1;
            
        } catch (Exception e) {
            System.out.println("catch");
            return 2;
            
        } finally {
            System.out.println("finally");
            return 3;
        }
```

查看上述代码编译后的字节码
```shell
public class com.shein.luban.common.TryCatchTest {
  public com.shein.luban.common.TryCatchTest();
    Code:
       0: aload_0
       1: invokespecial #1                  // Method java/lang/Object."<init>":()V
       4: return

  public static void main(java.lang.String[]);
    Code:
       0: invokestatic  #7                  // Method test:()I
       3: pop
       4: return

  public static int test();
    Code:
    ####################################### 执行try ######################################################
       0: getstatic     #13                 // Field java/lang/System.out:Ljava/io/PrintStream;
       3: ldc           #19                 // String try
       5: invokevirtual #21                 // Method java/io/PrintStream.println:(Ljava/lang/String;)V
       8: iconst_1

       ## 这里插入了finally中的代码块
      9: istore_0
      10: getstatic     #13                 // Field java/lang/System.out:Ljava/io/PrintStream;
      13: ldc           #27                 // String finally
      15: invokevirtual #21                 // Method java/io/PrintStream.println:(Ljava/lang/String;)V
      18: iconst_3
      19: ireturn

      ####################################### 执行catch ######################################################
      20: astore_0
      21: getstatic     #13                 // Field java/lang/System.out:Ljava/io/PrintStream;
      24: ldc           #31                 // String catch
      26: invokevirtual #21                 // Method java/io/PrintStream.println:(Ljava/lang/String;)V
      29: iconst_2

        ## 插入到catch中的finally代码块
      30: istore_1
      31: getstatic     #13                 // Field java/lang/System.out:Ljava/io/PrintStream;
      34: ldc           #27                 // String finally
      36: invokevirtual #21                 // Method java/io/PrintStream.println:(Ljava/lang/String;)V
      39: iconst_3
      40: ireturn

        ####################################### 执行finally ######################################################
      41: astore_2
      42: getstatic     #13                 // Field java/lang/System.out:Ljava/io/PrintStream;
      45: ldc           #27                 // String finally
      47: invokevirtual #21                 // Method java/io/PrintStream.println:(Ljava/lang/String;)V
      50: iconst_3
      51: ireturn


    ####################################### 异常表 ######################################################
    Exception table:
       from    to  target type
           0    10    20   Class java/lang/Exception
           0    10    41   any
          20    31    41   any
}

```

我们来逐步分析上述字节码，整个字节码按功能类型可以分为异常表，try主体，catch主体，finally主体。不过想要看懂上面的天书，还要学习几个简单的字节码指令。
字节码指令比较多，这里只介绍跟本文相关的指令

| 指令 | 内循环操作次数 |
| --- | --- |
| getstatic | 获取指定类的静态域，并将其值压入栈顶<br/> 一头雾水吧，这个咱不管，但是从上面代码大概能看出是获取PrintStream对象 |
| ldc | ldc指令用来将常量池中指定的常量放入操作数栈中,从注释看，这玩意就是获取要输出的字符串 |
| invokevirtual | 都invoke了，肯定是执行了方法。同理，注释中说是执行了PrintStream的println方法 |
| iconst_n | 将int类型的变量值n放到栈顶，类比，有lconst_n等 |
| istore_n/astore_n | 将栈顶的变量放到临时变量表中下标n的位置里 |
| ireturn | 将栈顶的变量出栈返回 |

#### 异常表
异常表有几个关键属性，分别如下：
  from: 从字节码哪里开始识别异常
  to:   到字节码哪里结束识别异常
  target: 遇到异常后，要跳转到哪里
  type: 识别的异常类型是啥

整体解释如下：
0~10: try主体代码，第一行第二行结合起来就是，如果在try中识别到了异常，如果类型是Exception就跳转到字节码第20行继续执行；如果类型不是Exception，那么就跳转到第41行执行。  
20行是catch开始的位置，41行是finally开始的位置。
那么就是如果代码识别的到的异常是Exception类型，就执行catch，否则就执行finally。是不是很简单？O(∩_∩)O~~


#### try代码块

```shell
####################################### 执行try ######################################################
       0: getstatic     #13                 // Field java/lang/System.out:Ljava/io/PrintStream;
       3: ldc           #19                 // String try
       5: invokevirtual #21                 // Method java/io/PrintStream.println:(Ljava/lang/String;)V
       8: iconst_1

       ## 这里插入了finally中的代码块
      9: istore_0
      10: getstatic     #13                 // Field java/lang/System.out:Ljava/io/PrintStream;
      13: ldc           #27                 // String finally
      15: invokevirtual #21                 // Method java/io/PrintStream.println:(Ljava/lang/String;)V
      18: iconst_3
      19: ireturn
```
0~5输出字符串try,第8行将int 1 压入栈顶。第9行将栈顶的1放到临时变量表的0号下标。然后又执行finally里面的System.out.println，18行将int值3压入栈顶，然后执行ireturn，将3弹出栈顶并返回。  
人话讲就是：上面代码没有抛出异常，最终会返回finally的返回值。


#### catch代码块

```shell
####################################### 执行catch ######################################################
      20: astore_0
      21: getstatic     #13                 // Field java/lang/System.out:Ljava/io/PrintStream;
      24: ldc           #31                 // String catch
      26: invokevirtual #21                 // Method java/io/PrintStream.println:(Ljava/lang/String;)V
      29: iconst_2

        ## 插入到catch中的finally代码块
      30: istore_1
      31: getstatic     #13                 // Field java/lang/System.out:Ljava/io/PrintStream;
      34: ldc           #27                 // String finally
      36: invokevirtual #21                 // Method java/io/PrintStream.println:(Ljava/lang/String;)V
      39: iconst_3
      40: ireturn
```
20行标识将栈顶的变量值放到临时变量表的0号位置，然后输入字符串catch，接下来将int元素2放到栈顶。随后，代码再次进入finally代码块中，先将栈顶的元素2放到临时变量表的1号位置。输入finally字符串，将int元素3压入栈顶，并执行ireturn弹出栈顶元素3后返回。  
如果catch执行过程中没有抛出异常，那么代码执行完毕后。局部变量栈中只有2，局部变量表是[null, 2]。

#### finally代码块

```shell
####################################### 执行finally ######################################################
      41: astore_2
      42: getstatic     #13                 // Field java/lang/System.out:Ljava/io/PrintStream;
      45: ldc           #27                 // String finally
      47: invokevirtual #21                 // Method java/io/PrintStream.println:(Ljava/lang/String;)V
      50: iconst_3
      51: ireturn
```
这个代码不做过多解释了，执行完后弹出栈顶元素3并返回。   
如果finally执行没有异常，那么在本地变量表可能是[1, null, null]或者[null, 2, null]


