# kyleluther.github.io

Repo for my website, built using Jekyll.

Normally, Jekyll sets your posts to be the default page. To set the contents of about.md to be the main landing page, I remove the index.md file and use [jekyll-redirect-from](https://github.com/jekyll/jekyll-redirect-from) to redirect the contents of about.md to the main homepage. Addtionally I create a posts.md file which uses layout _home_ so it contains all posts i have written
